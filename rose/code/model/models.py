# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import GPy
import numpy as np
import gpytorch
import math, copy
import os

# Define the device at the top of the file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cpu_state_dict(obj):
    # Works for nn.Module and any object with state_dict()
    return {k: v.detach().cpu().clone() for k, v in obj.state_dict().items()}

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0, minimize=True):
        """
        patience: consecutive epochs allowed without sufficient improvement
        min_delta: required improvement to reset patience (best - loss > min_delta)
        minimize: True for minimizing loss (default); False if maximizing a metric
        """
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.minimize = bool(minimize)

        self.best_loss = float('inf') if minimize else -float('inf')
        self.num_bad_epochs = 0
        self.best_epoch = -1
        self.best_states = {}   # dict[name -> state_dict]
        self._has_best = False

    @property
    def has_best(self):
        return self._has_best

    def _is_improved(self, value):
        if self.minimize:
            return (self.best_loss - value) > self.min_delta
        else:
            return (value - self.best_loss) > self.min_delta

    def step(self, loss_value, epoch_idx, **named_modules):
        """
        Check improvement and (if improved) store CPU copies of states.

        Usage:
          stopper.step(avg_loss, epoch, model=model)                     # BNN
          stopper.step(loss_val, epoch, model=model, likelihood=lik)     # GPR
        """
        improved = self._is_improved(float(loss_value))
        if improved:
            self.best_loss = float(loss_value)
            self.best_epoch = int(epoch_idx)
            self.num_bad_epochs = 0
            # Save states of all provided modules (model, likelihood, …)
            self.best_states = {
                name: _cpu_state_dict(mod) for name, mod in named_modules.items() if mod is not None
            }
            self._has_best = True
            return False, True   # stop=False, saved_best=True
        else:
            self.num_bad_epochs += 1
            stop = self.num_bad_epochs >= self.patience
            return stop, False

    def load_best(self, **named_modules):
        """
        Restore best states into provided modules by name.
        Example:
          stopper.load_best(model=model)                          # BNN
          stopper.load_best(model=model, likelihood=likelihood)   # GPR
        """
        if not self._has_best:
            return False
        restored_any = False
        for name, mod in named_modules.items():
            if mod is not None and name in self.best_states:
                mod.load_state_dict(self.best_states[name])
                restored_any = True
        return restored_any

    def save_best(self, path):
        """
        Save best states & metadata to disk (creates parent dirs if needed).
        """
        if not self._has_best:
            return False
        _ensure_dir(path)
        payload = {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "best_states": self.best_states,
            "minimize": self.minimize,
            "min_delta": self.min_delta,
            "patience": self.patience,
        }
        torch.save(payload, path)
        return True

    @staticmethod
    def load_from(path):
        """
        Load a saved EarlyStopper snapshot (for inspection or resuming).
        """
        ckpt = torch.load(path, map_location="cpu")
        es = EarlyStopper(patience=ckpt.get("patience", 10),
                          min_delta=ckpt.get("min_delta", 0.0),
                          minimize=ckpt.get("minimize", True))
        es.best_loss = ckpt["best_loss"]
        es.best_epoch = ckpt["best_epoch"]
        es.best_states = ckpt["best_states"]
        es._has_best = True
        return es




# Bayesian Neural Network (BNN)
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_init_std, log_std_init_mean, log_std_init_std):
        super().__init__()
        self.mean = nn.Parameter(torch.randn(out_features, in_features) * weight_init_std)
        self.log_std = nn.Parameter(torch.randn(out_features, in_features) * log_std_init_std + log_std_init_mean)
        self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        weights = self.mean + torch.randn_like(self.log_std) * torch.exp(self.log_std)
        return F.linear(x, weights, self.b)

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, n_features, hidden_layers, weight_init_std, log_std_init_mean, log_std_init_std, log_std_clamp):
        super().__init__()
        layers = []
        prev_size = n_features
        for size in hidden_layers:
            layers.extend([BayesianLinear(prev_size, size, weight_init_std, log_std_init_mean, log_std_init_std), nn.ReLU()])
            prev_size = size
        layers.append(BayesianLinear(prev_size, 2, weight_init_std, log_std_init_mean, log_std_init_std))
        self.layers = nn.Sequential(*layers)
        self.log_std_clamp = log_std_clamp

    def forward(self, x):
        output = self.layers(x)
        return torch.distributions.Normal(output[:, 0], torch.exp(output[:, 1].clamp(*self.log_std_clamp)))

def NLL_loss_bnn(targets, distribution):
    return -distribution.log_prob(targets).mean()

def hybrid_loss_bnn(targets, distribution, beta=0.1, gamma=0.01):
    nll = -distribution.log_prob(targets).mean()
    mean_pred = distribution.mean
    variance_pred = distribution.variance
    mse = F.mse_loss(mean_pred, targets)
    var_reg = torch.relu(0.1 - variance_pred.mean())
    return gamma*nll + mse + gamma*var_reg


def train_bnn(model, train_loader, epochs, learning_rate, grad_clip_norm, min_delta=1e-4, patience=10, checkpoint_path=None, load_best_on_end=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    device = next(model.parameters()).device
    stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = target.flatten()
            optimizer.zero_grad()
            distribution = model(data)
            loss = NLL_loss_bnn(target, distribution)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - train loss: {avg_loss:.6f}")
        stop, saved = stopper.step(avg_loss, epoch, model=model)
        if saved and checkpoint_path is not None:
            stopper.save_best(checkpoint_path)
        if stop:
            print(f"Early stop at {epoch+1}; best @ {stopper.best_epoch+1} loss={stopper.best_loss:.6f}")
            break
    if load_best_on_end and stopper.best_states is not None:
        stopper.load_best(model=model)
    return {
        "train_loss_history": history,
        "best_epoch": stopper.best_epoch,
        "best_train_loss": stopper.best_loss,
        "stopped_early": (len(history) - 1) != (epochs - 1)
    }
             

def predict_bnn(model, input_data, n_samples=100):
    model.eval()
    with torch.no_grad():
        samples = torch.stack([model(input_data).sample() for _ in range(n_samples)])
    return samples.mean(0), samples.std(0)


# Gaussian Process Regression (GPR)
def train_gpr(xtrain, ytrain, kernel_variance, kernel_lengthscale, white_kernel_variance, max_iterations):
    kernel = GPy.kern.Matern32(input_dim=xtrain.shape[1], variance=kernel_variance, lengthscale=kernel_lengthscale)
    kernel += GPy.kern.White(xtrain.shape[1], variance=white_kernel_variance)
    model = GPy.models.GPRegression(xtrain, ytrain, kernel)
    model.optimize(max_iters=max_iterations)
    return model

def predict_gpr(model, xtest):
    mean, variance = model.predict(xtest)
    std = np.sqrt(variance)
    return mean, std


#GPR using GPyTorch on GPU
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_exactgpr(xtrain, ytrain, epochs, learning_rate, min_delta=1e-4, patience=10, checkpoint_path=None, load_best_on_end=True):
    ytrain = ytrain.squeeze(-1)  # ensure shape [N]
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(xtrain, ytrain, likelihood).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = model(xtrain)
        loss = -mll(output, ytrain).mean()
        loss_value = loss.item()
        loss.backward()
        optimizer.step()

        history.append(loss_value)
        print(f"Epoch {epoch+1}/{epochs} - train loss: {loss_value:.6f}")
        stop, saved = stopper.step(loss_value, epoch, model=model, likelihood=likelihood)
        #if saved and checkpoint_path is not None:
            #stopper.save_best(checkpoint_path)
        if stop:
            print(f"Early stop at {epoch+1}; best @ {stopper.best_epoch} loss={stopper.best_loss:.6f}")
            break
    #if load_best_on_end and stopper.best_states is not None:
        #stopper.load_best(model=model, likelihood=likelihood)
    return model, likelihood


def predict_exactgpr(model, xtest):
    model.eval()
    with torch.no_grad():
        f_preds = model(xtest)
    return f_preds.mean, f_preds.variance.clamp_min(1e-6)
    



# Monte Carlo Dropout
class DropoutModel(nn.Module):
    def __init__(self, n_bits, n_1, n_2, n_3, dropout_rate):
        super(DropoutModel, self).__init__()
        self.fc1 = nn.Linear(n_bits, n_1)
        self.fc2 = nn.Linear(n_1, n_2)
        self.fc3 = nn.Linear(n_2, n_3)
        self.fc4 = nn.Linear(n_3, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, apply_dropout=True):
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if apply_dropout else x
        x = F.relu(self.fc2(x))
        x = self.dropout(x) if apply_dropout else x
        x = F.relu(self.fc3(x))
        x = self.dropout(x) if apply_dropout else x
        return self.fc4(x)

def train_mcd(model, train_loader, epochs, learning_rate, min_delta=1e-4, patience=10, checkpoint_path=None, load_best_on_end=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - train loss: {avg_loss:.6f}")
        stop, saved = stopper.step(avg_loss, epoch, model=model)
        if saved and checkpoint_path is not None:
            stopper.save_best(checkpoint_path)
        if stop:
            print(f"Early stop at {epoch+1}; best @ {stopper.best_epoch+1} loss={stopper.best_loss:.6f}")
            break
    if load_best_on_end and stopper.best_states is not None:
        stopper.load_best(model=model)
    return {
        "train_loss_history": history,
        "best_epoch": stopper.best_epoch,
        "best_train_loss": stopper.best_loss,
        "stopped_early": (len(history) - 1) != (epochs - 1)
    }


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def predict_mcd(model, input_data, n_samples):
    model.eval()
    enable_dropout(model)
    #all_predictions = []

    with torch.no_grad():
        samples = torch.stack([model(input_data, apply_dropout=True) for _ in range(n_samples)])
    return samples.mean(0), samples.std(0)