import argparse
import os, time
import yaml
import torch
import numpy as np
from model.models import BayesianNeuralNetwork, train_bnn, train_gpr, predict_bnn, predict_gpr
from torch.utils.data import TensorDataset, DataLoader
from utils import calculate_rmse
import json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--iteration", type=int, default=1)
    p.add_argument("--pipeline_dir", required=True)
    args = p.parse_args()

#    args = argparse.Namespace(
#            config    = '/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/config/gpr.yaml',
#            iteration = 1,
#            pipeline_dir  = '/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_01/'
#            )

    cfg = yaml.safe_load(open(args.config))
    out_dir = os.path.join(args.pipeline_dir, f"iter_{args.iteration:03d}")
    test_dir = os.path.join(args.pipeline_dir, f"iter_001")

    x_train  = np.load(os.path.join(out_dir,  'scaled_x_train.npy'),  mmap_mode='r').copy() 
    y_train  = np.load(os.path.join(out_dir,  'scaled_y_train.npy'),  mmap_mode='r').copy()
    x_test   = np.load(os.path.join(test_dir, 'scaled_x_test.npy'),   mmap_mode='r').copy()
    y_test   = np.load(os.path.join(test_dir, 'scaled_y_test.npy'),   mmap_mode='r').copy()

    metrics = {
        'rmse': None,
        'training_size': None,
        'test_size': None,
        'std': None
    }

    if cfg["model"] == "gpr":
        start_time = time.time()
        model = train_gpr(x_train, y_train,
                          cfg["kernel_variance"],
                          cfg["kernel_lengthscale"],
                          cfg["white_kernel_variance"],
                          cfg["max_iterations"])
        train_time = time.time() - start_time
        model.pickle(os.path.join(out_dir, "model.pkl"))

        mean, std = predict_gpr(model, x_test)
        rmse = calculate_rmse(y_test, mean)
        metrics['rmse'] = rmse
        metrics['training_size'] = len(x_train)
        metrics['test_size'] = len(x_test)
        metrics['std'] = np.mean(std)

        print(f"Training time: {train_time:.2f} seconds | RMSE: {rmse:.4f}")
        print(f"Prediction stats: Mean={np.mean(mean):.4f} ± Std={np.mean(std):.4f}")

    elif cfg["model"] == "bnn":
        n_features = x_train.shape[1]
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)
        x_test  = torch.FloatTensor(x_test)
        y_test  = torch.FloatTensor(y_test)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BayesianNeuralNetwork(
            n_features,
            cfg["hidden_layers"],
            cfg["weight_init_std"],
            cfg["log_std_init_mean"],
            cfg["log_std_init_std"],
            tuple(cfg["log_std_clamp"])
        ).to(device)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

        start_time = time.time()
        train_bnn(model,
                  train_loader,
                  cfg["bnn_epochs"],
                  cfg["learning_rate"],
                  cfg["grad_clip_norm"])
        train_time = time.time() - start_time

        with torch.no_grad():
            test_preds, test_std = predict_bnn(model, x_test.to(device), n_samples=cfg["n_mc_samples"]) # FIXME! What is n_mc_samples?

        # Calculate RMSE
        rmse = calculate_rmse(y_test.numpy(), test_preds.cpu().numpy())
        metrics['rmse'] = rmse
        metrics['training_size'] = len(x_train)
        metrics['test_size'] = len(x_test)
        metrics['std'] = test_std.mean().item()

        print(f"Training time: {train_time:.2f} seconds | RMSE: {rmse:.4f}")
        print(f"Prediction stats: Mean={test_preds.mean().item():.4f} ± Std={test_std.mean().item():.4f}")
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    else:
        raise Exception(f"Model of {cfg['model']} currently not supported in {os.path.basename(__file__)}!!!")

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    print(f"[train] done iteration {args.iteration}")

if __name__=="__main__":
    main()
