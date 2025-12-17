import os
import asyncio

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# ROSE imports for custom Learner approach
from rose.learner import Learner
from radical.asyncflow import WorkflowEngine
from radical.asyncflow import ConcurrentExecutionBackend
from radical.asyncflow.logging import init_default_logger
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


logger = logging.getLogger(__name__)

@dataclass
class LearnerResult:
    """Result from a learner including performance metrics and timing"""
    learner_id: int
    performance_score: float
    completion_time: float
    model_data: Dict[str, Any]
    success: bool
    killed: bool = False


async def simulation_task(*args, **kwargs) -> Dict[str, Any]:
    """Actual simulation task - exact logic from simulation.py"""
    import os
    import pickle
    import numpy as np
    import joblib
    from sklearn.utils import shuffle
    from utils import preprocess_inputdata, compute_peak_density, preprocess_inputdata_based_on_bins_positive, expand_input_output, collapse_input_output
    
    iteration = kwargs.get("iteration", 0)
    input_data_dir = kwargs.get("input_data_dir", "")
    seed = kwargs.get("seed", 42)
    
    print(f"Running simulation for iteration {iteration}")
    
    # Exact logic from simulation.py - Load preprocessed data
    # with open(os.path.join(input_data_dir, 'data_dump_density_preprocessed_train.pk'), 'rb') as handle:
    #     processed_all_data_preprocessed_train = pickle.load(handle)
    # with open(os.path.join(input_data_dir, 'data_dump_density_preprocessed_test.pk'), 'rb') as handle:
    #     processed_all_data_preprocessed_test = pickle.load(handle)

    with open(os.path.join(input_data_dir, 'data_dump_density_preprocessed_train_V3.pk'), 'rb') as handle:
        processed_all_data_preprocessed_train = pickle.load(handle)
    with open(os.path.join(input_data_dir, 'data_dump_density_preprocessed_test_V3.pk'), 'rb') as handle:
        processed_all_data_preprocessed_test = pickle.load(handle)

    # Reduce training set size by randomly excluding N data
    np.random.seed(seed)
    index_ = np.random.choice(len(processed_all_data_preprocessed_train.keys()), 3545, replace=False)
    excluded_index_ = np.delete(np.arange(0, len(processed_all_data_preprocessed_train.keys())), index_)
    
    train_ = {}
    exclude_ = {}

    for index in index_:
        exclude_[list(processed_all_data_preprocessed_train.keys())[index]] = processed_all_data_preprocessed_train[list(processed_all_data_preprocessed_train.keys())[index]]
    for index in excluded_index_:
        train_[list(processed_all_data_preprocessed_train.keys())[index]] = processed_all_data_preprocessed_train[list(processed_all_data_preprocessed_train.keys())[index]]

    # # Preprocess data to density output (NX1004)
    # input_data, output, errors, z_data = preprocess_inputdata(train_)
    # input_data_remain, output_remain, errors_remain, z_data_remain = preprocess_inputdata(exclude_)
    # input_data_test, output_test_raw, errors_test_raw, z_data_test = preprocess_inputdata(processed_all_data_preprocessed_test)

    # # Convert to peak density output (NX1)
    # output_train, errors_train = compute_peak_density(input_data, output, errors, z_data)
    # output_train_remain, errors_train_remain = compute_peak_density(input_data_remain, output_remain, errors_remain, z_data_remain)
    # output_test, errors_test = compute_peak_density(input_data_test, output_test_raw, errors_test_raw, z_data_test)

    input_data, output, errors, z_data = preprocess_inputdata_based_on_bins_positive(train_)
    input_data_remain, output_train_remain, errors_train_remain, z_data_remain = preprocess_inputdata_based_on_bins_positive(exclude_)
    input_data_test, output_test, errors_test, z_data_test = preprocess_inputdata_based_on_bins_positive(processed_all_data_preprocessed_test)

    # Cross validation - split ranges 0.8 to 1
    train_test_split = 1

    # input_data_suff, output_suff, errors_suff, z_data_shuff = shuffle(
    #     input_data, output_train, errors_train, z_data, random_state=seed
    # )

    input_data_suff, output_suff, errors_suff, z_data_shuff = shuffle(
        input_data, output, errors, z_data, random_state=seed
    )

    train_test_split_ = int(input_data_suff.shape[0] * train_test_split)
    
    x_train = input_data_suff[0:train_test_split_]
    x_test = input_data_suff[train_test_split_:]
    y_train = output_suff[0:train_test_split_]
    y_test = output_suff[train_test_split_:]
    error_train = errors_suff[0:train_test_split_]
    error_test = errors_suff[train_test_split_:]

    print("Train input: ", x_train.shape)
    print("Train Output", y_train.shape)
    print("Test input: ", x_test.shape)
    print("Test Output", y_test.shape)

    # Load scalers and transform data
    scaler = joblib.load(os.path.join(input_data_dir, 'scaler_new.pkl'))
    scaled_x_train = scaler.transform(x_train)
    scaled_x_test = scaler.transform(input_data_test)

    # scaler_y = joblib.load(os.path.join(input_data_dir, 'minmax_scaler_peak_label.joblib'))
    # scaler_error = joblib.load(os.path.join(input_data_dir, 'minmax_scaler_peak_error.joblib'))
    scaler_y = joblib.load(os.path.join(input_data_dir, 'minmax_scaler_40_labels.joblib'))
    scaler_error = joblib.load(os.path.join(input_data_dir, 'minmax_scaler_40_errors.joblib'))

    scaled_y_train = scaler_y.transform(y_train)
    scaled_y_test = scaler_y.transform(output_test)
    scaled_error_train = scaler_error.transform(error_train)
    scaled_error_test = scaler_error.transform(errors_test)

    scaled_x_remain = scaler.transform(input_data_remain)
    scaled_y_remain = scaler_y.transform(output_train_remain)
    scaled_error_remain = scaler_error.transform(errors_train_remain)

    # Augment data 
    scaled_x_train = np.asarray(scaled_x_train, dtype=np.float64)
    scaled_y_train = np.asarray(scaled_y_train, dtype=np.float64)
    X_expanded_train, Y_expanded_train = expand_input_output(scaled_x_train, scaled_y_train)
    print("Expanded input shape:", X_expanded_train.shape)   # (n, 8)
    print("Expanded output shape:", Y_expanded_train.shape)  # (m, 1)

    scaled_x_test = np.asarray(scaled_x_test, dtype=np.float64)
    scaled_y_test = np.asarray(scaled_y_test, dtype=np.float64)
    X_expanded_test, Y_expanded_test = expand_input_output(scaled_x_test, scaled_y_test)
    print("Expanded input shape:", X_expanded_test.shape)   # (n, 8)
    print("Expanded output shape:", Y_expanded_test.shape)  # (m, 1)

    scaled_x_remain = np.asarray(scaled_x_remain, dtype=np.float64)
    scaled_y_remain = np.asarray(scaled_y_remain, dtype=np.float64)
    X_expanded_remain, Y_expanded_remain = expand_input_output(scaled_x_remain, scaled_y_remain)
    print("Expanded input shape:", X_expanded_remain.shape)   # (n, 8)
    print("Expanded output shape:", Y_expanded_remain.shape)  # (m, 1)

    # Return dictionary instead of MLData object
    # return {
    #     "x_train": scaled_x_train,
    #     "y_train": scaled_y_train,
    #     "x_test": scaled_x_test,
    #     "y_test": scaled_y_test,
    #     "x_remain": scaled_x_remain,
    #     "y_remain": scaled_y_remain,
    #     "error_train": scaled_error_train,
    #     "error_test": scaled_error_test,
    #     "error_remain": scaled_error_remain,
    #     "metadata": {
    #         "iteration": iteration,
    #         "train_shape": x_train.shape,
    #         "test_shape": x_test.shape,
    #         "remain_shape": scaled_x_remain.shape,
    #         "scalers": {
    #             "scaler": scaler,
    #             "scaler_y": scaler_y,
    #             "scaler_error": scaler_error
    #         }
    #     }
    # }

    return {
        "x_train": X_expanded_train,
        "y_train": Y_expanded_train,
        "x_test": X_expanded_test,
        "y_test":  Y_expanded_test,
        "x_remain": X_expanded_remain,
        "y_remain": Y_expanded_remain,
        "error_train": scaled_error_train,
        "error_test": scaled_error_test,
        "error_remain": scaled_error_remain,
        "metadata": {
            "iteration": iteration,
            "train_shape": x_train.shape,
            "test_shape": x_test.shape,
            "remain_shape": scaled_x_remain.shape,
            "scalers": {
                "scaler": scaler,
                "scaler_y": scaler_y,
                "scaler_error": scaler_error
            }
        }
    }

async def training_task(*args, **kwargs) -> Dict[str, Any]:
    """Actual training task - NO timeout logic here anymore"""
    import time
    import yaml
    import torch
    import asyncio
    import numpy as np
    from model.models import EarlyStopper, BayesianNeuralNetwork, train_bnn, predict_bnn, train_exactgpr, predict_exactgpr, DropoutModel, train_mcd, predict_mcd, enable_dropout
    import gpytorch
    from torch.utils.data import TensorDataset, DataLoader
    from utils import calculate_rmse, calculate_r2, calculate_spearman
    import math, copy
    
    learner_id = kwargs.get("learner_id", 0)
    iteration = kwargs.get("iteration", 0)
    config_path = kwargs.get("config_path", "")
    ml_data = kwargs.get("ml_data")
    
    if not isinstance(ml_data, dict):
        raise ValueError("ml_data must be a dictionary")
    
    print(f"Learner {learner_id} starting training for iteration {iteration}")
    
    start_time = time.time()
    
    # Load config - exact logic from train.py
    cfg = yaml.safe_load(open(config_path)) if config_path else {"model": "gpr"}
    
    # Use data directly from memory
    x_train = ml_data["x_train"].copy()  # Make copies to avoid modifying original data
    y_train = ml_data["y_train"].copy()
    x_test = ml_data["x_test"].copy()
    y_test = ml_data["y_test"].copy()

    metrics = {
        'rmse': None,
        'training_size': len(x_train),
        'test_size': len(x_test),
        'std': None,
        'R2': None,
        'Spearman': None
    }

    # NO timeout logic here - just do the training
    if cfg["model"] == "gpr":
        # Exact GPR training logic from train.py
        n_features = x_train.shape[1]
        x_train_tensor = torch.FloatTensor(x_train)
        y_train_tensor = torch.FloatTensor(y_train)
        x_test_tensor = torch.FloatTensor(x_test)
        y_test_tensor = torch.FloatTensor(y_test)
        train_start = time.time()
        with gpytorch.settings.fast_computations(covar_root_decomposition=False,
                                                log_prob=False,
                                                solves=False), \
            gpytorch.settings.max_preconditioner_size(0), \
            gpytorch.settings.skip_logdet_forward(False), \
            gpytorch.settings.skip_posterior_variances(False):
            device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()           # frees unreferenced GPU memory
            model, likelihood = train_exactgpr(x_train_tensor.to(device), y_train_tensor.to(device),
                                cfg["epochs"],
                                cfg["learning_rate"],
                                cfg["min_delta"],
                                cfg["patience"],
                                cfg["checkpoint_path"])
        train_time = time.time() - train_start

        mean, std = predict_exactgpr(model, x_test_tensor.to(device))
        rmse = calculate_rmse(y_test, mean.cpu().numpy())
        r2 = calculate_r2(y_test, mean.cpu().numpy())
        spearman = calculate_spearman(y_test, mean.cpu().numpy(), std.cpu().numpy())
        metrics['rmse'] = rmse
        metrics['std'] = std.detach().mean().item()
        metrics['R2'] = r2
        metrics['Spearman'] = spearman
        print(f"Training time: {train_time:.2f} seconds | RMSE: {rmse:.4f}")
        print(f"Prediction stats: Mean={mean.detach().mean().item():.4f} ± Std={std.detach().mean().item():.4f}")

        training_result = {"model": model, "rmse": rmse, "std": std.detach().mean().item(), "R2": r2, "Spearman": spearman}

    elif cfg["model"] == "bnn":
        # Exact BNN training logic from train.py
        n_features = x_train.shape[1]
        x_train_tensor = torch.FloatTensor(x_train)
        y_train_tensor = torch.FloatTensor(y_train)
        x_test_tensor = torch.FloatTensor(x_test)
        y_test_tensor = torch.FloatTensor(y_test)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BayesianNeuralNetwork(
            n_features,
            cfg["hidden_layers"],
            cfg["weight_init_std"],
            cfg["log_std_init_mean"],
            cfg["log_std_init_std"],
            tuple(cfg["log_std_clamp"])
        ).to(device)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

        train_start = time.time()
        train_bnn(model,
                    train_loader,
                    cfg["bnn_epochs"],
                    cfg["learning_rate"],
                    cfg["grad_clip_norm"],
                    cfg["min_delta"],
                    cfg["patience"],
                    cfg["checkpoint_path"])
        train_time = time.time() - train_start

        with torch.no_grad():
            test_preds, test_std = predict_bnn(model, x_test_tensor.to(device), n_samples=cfg["n_mc_samples"])

        # Calculate RMSE
        rmse = calculate_rmse(y_test, test_preds.cpu().numpy())
        r2 = calculate_r2(y_test, test_preds.cpu().numpy())
        spearman = calculate_spearman(y_test, test_preds.cpu().numpy(), test_std.cpu().numpy())
        metrics['rmse'] = rmse
        metrics['std'] = test_std.mean().item()
        metrics['R2'] = r2
        metrics['Spearman'] = spearman

        print(f"Training time: {train_time:.2f} seconds | RMSE: {rmse:.4f}")
        print(f"Prediction stats: Mean={test_preds.mean().item():.4f} ± Std={test_std.mean().item():.4f}")

        training_result = {"model": model, "rmse": rmse, "std": test_std.mean().item(), "R2": r2, "Spearman": spearman}

    elif cfg["model"] == "mcd":
        n_features = x_train.shape[1]
        x_train_tensor = torch.FloatTensor(x_train)
        y_train_tensor = torch.FloatTensor(y_train)
        x_test_tensor = torch.FloatTensor(x_test)
        y_test_tensor = torch.FloatTensor(y_test)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DropoutModel(
            n_features,
            cfg["hidden_layers_0"],
            cfg["hidden_layers_1"],
            cfg["hidden_layers_2"],
            cfg["dropout"]).to(device)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

        train_start = time.time()
        train_mcd(model,
                    train_loader,
                    cfg["mcd_epochs"],
                    cfg["learning_rate"],
                    cfg["min_delta"],
                    cfg["patience"],
                    cfg["checkpoint_path"])
        train_time = time.time() - train_start

        with torch.no_grad():
            test_preds, test_std = predict_mcd(model, x_test_tensor.to(device), n_samples=cfg["n_mc_samples"])

        # Calculate RMSE
        rmse = calculate_rmse(y_test, test_preds.cpu().numpy())
        r2 = calculate_r2(y_test, test_preds.cpu().numpy())
        spearman = calculate_spearman(y_test, test_preds.cpu().numpy(), test_std.cpu().numpy())
        metrics['rmse'] = rmse
        metrics['std'] = test_std.mean().item()
        metrics['R2'] = r2
        metrics['Spearman'] = spearman

        print(f"Training time: {train_time:.2f} seconds | RMSE: {rmse:.4f}")
        print(f"Prediction stats: Mean={test_preds.mean().item():.4f} ± Std={test_std.mean().item():.4f}")

        training_result = {"model": model, "rmse": rmse, "std": test_std.mean().item(), "R2": r2, "Spearman": spearman}   
    else:
        raise Exception(f"Model of {cfg['model']} currently not supported!")

    completion_time = time.time() - start_time

    # Performance is inverse of RMSE (lower RMSE = higher performance)
    #performance_score = max(0.1, 1.0 - training_result["rmse"])
    performance_score = 0.9*training_result["R2"] + 0.1*training_result["Spearman"]

    result = {
        "learner_id": learner_id,
        "performance_score": performance_score,
        "completion_time": completion_time,
        "iteration": iteration,
        "model_params": {
            "rmse": training_result["rmse"],
            "training_size": metrics['training_size'],
            "test_size": metrics['test_size'],
            "std": training_result["std"],
            "R2": training_result["R2"],
            "Spearman": training_result["Spearman"]
        },
        "training_metadata": {
            "model_type": cfg["model"],
            "config": cfg
        },
        "trained_model": training_result["model"],  # Store actual trained model
        "killed": False,
        "success": True
    }
    
    print(f"Learner {learner_id} completed in {completion_time:.1f}s (RMSE: {training_result['rmse']:.4f}) (R2: {training_result['R2']:.4f}) (performance_score: {performance_score:.4f}) (R2: {training_result['R2']:.4f}) (Spearman: {training_result['Spearman']:.4f})")
    return result


async def active_learn_task(*args, **kwargs) -> Dict[str, Any]:
    """Actual active learning task - exact logic from active.py"""
    import yaml
    import torch
    import numpy as np
    import gpytorch
    from model.models import EarlyStopper, BayesianNeuralNetwork, train_bnn, predict_bnn, train_exactgpr, predict_exactgpr, predict_mcd
    
    selected_learner = kwargs.get("selected_learner")
    iteration = kwargs.get("iteration", 0)
    config_path = kwargs.get("config_path", "")
    ml_data = kwargs.get("ml_data")
    n_new_samples = kwargs.get("n_new_samples", 5)

    if not selected_learner:
        return None

    if not isinstance(ml_data, dict):
        raise ValueError("ml_data must be a dictionary")
    
    print(f"Generating Data X from learner {selected_learner['learner_id']} for iteration {iteration + 1}")
    
    # Load config - exact logic from active.py
    cfg = yaml.safe_load(open(config_path)) if config_path else {"model": "gpr"}
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use data directly from memory - make copies to avoid modifying original
    x_train = ml_data["x_train"].copy()
    y_train = ml_data["y_train"].copy()
    x_remain = ml_data["x_remain"].copy() if ml_data.get("x_remain") is not None else None
    y_remain = ml_data["y_remain"].copy() if ml_data.get("y_remain") is not None else None

    if x_remain is None or y_remain is None:
        print("No remaining data available for active learning")
        return ml_data  # Return current data unchanged

    # Get trained model from selected learner (in-memory instead of file loading)
    trained_model = selected_learner.get('trained_model')
    if not trained_model:
        print("Warning: No trained model found, using random selection")
        idx = np.random.choice(len(x_remain), min(n_new_samples, len(x_remain)), replace=False)
    else:
        # Exact active learning logic from active.py - but using in-memory model
        if cfg["model"] == "gpr":
            # Original: model = GPy.load(os.path.join(in_dir, "model.pkl"))
            # Now: use the trained_model directly (it's already a GPy model from training task)
            model = trained_model
            torch.save(model.state_dict(), f"saved_model/gpr_model_state_{iteration}.pt")        #save model to the file path: saved_model/..
            _, std_remain = predict_exactgpr(model, torch.FloatTensor(x_remain).to(device))
            idx = np.argsort(std_remain.cpu().numpy().flatten())[-n_new_samples:]
            
        elif cfg["model"] == "bnn":
            # Original: state = torch.load(os.path.join(in_dir, "model.pt"), map_location=device)
            #           model.load_state_dict(state)
            # Now: use the trained_model directly (it's already a trained BNN from training task)
            n_features = x_train.shape[1]
            model = trained_model  # This is already the trained BNN model
            torch.save(model.state_dict(), f"saved_model/bnn_model_state_{iteration}.pt")        #save model to the file path: saved_model/..
            model.to(device).eval()
            _, std_remain = predict_bnn(model, torch.FloatTensor(x_remain).to(device), n_samples=cfg["n_mc_samples"])
            idx = np.argsort(std_remain.cpu().numpy())[-n_new_samples:]
        elif cfg["model"] == "mcd":
            n_features = x_train.shape[1]
            model = trained_model 
            torch.save(model.state_dict(), f"saved_model/mcd_model_state_{iteration}.pt")        #save model to the file path: saved_model/..
            model.to(device).eval()
            _, std_remain = predict_mcd(model, torch.FloatTensor(x_remain).to(device), n_samples=cfg["n_mc_samples"])
            idx = np.argsort(std_remain.cpu().numpy().flatten())[-n_new_samples:]         
        else:
            raise Exception(f"Model of {cfg['model']} currently not supported in active learning!")

    # Update training data with selected samples - exact logic from active.py
    x_train_new = np.vstack([x_train, x_remain[idx]])
    y_train_new = np.vstack([y_train, y_remain[idx]])
    
    # Remove selected samples from remaining data
    mask = np.ones(len(x_remain), bool)
    mask[idx] = False
    x_remain_new = x_remain[mask]
    y_remain_new = y_remain[mask]
    
    # Handle error data if available
    # error_train_new = ml_data.get("error_train")
    # error_remain_new = ml_data.get("error_remain")
    # if error_train_new is not None and error_remain_new is not None:
    #     error_train_new = np.vstack([error_train_new, error_remain_new[idx]])
    #     error_remain_new = error_remain_new[mask]

    print(f"Selected {len(idx)} new samples. Training set: {len(x_train)} -> {len(x_train_new)}")
    np.save(f"saved_data/scaled_x_train_{iteration}.npy",  x_train_new)
    np.save(f"saved_data/scaled_y_train_{iteration}.npy",  y_train_new)

    # Return updated dictionary
    return {
        "x_train": x_train_new,
        "y_train": y_train_new,
        "x_test": ml_data["x_test"],  # Test data remains the same
        "y_test": ml_data["y_test"],
        "x_remain": x_remain_new,
        "y_remain": y_remain_new,
        "error_train": None, #error_train_new,
        "error_test": None, #ml_data.get("error_test"),  # Test error remains the same
        "error_remain": None, # error_remain_new,
        "metadata": {
            **ml_data.get("metadata", {}),
            "selected_samples": len(idx),
            "new_training_size": len(x_train_new),
            "remaining_samples": len(x_remain_new),
            "source_learner": selected_learner['learner_id'],
            "iteration": iteration + 1
        }
    }





class SynchronizedLearner(Learner):
    
    def __init__(self, asyncflow, alpha: float = 1.5, max_iterations: int = 3, n_new_samples=5):
        """
        Initialize the synchronized learning workflow
        Args:
            alpha: Time multiplier for kill threshold (aT_min) - REMOVED t_min parameter
            max_iterations: Maximum number of iterations to run
            n_new_samples: Number of new samples to be selected for each new iteration
        """
        self.alpha = alpha
        # t_min is now calculated dynamically per iteration - no longer a fixed parameter
        self.max_iterations = max_iterations
        self.n_new_samples = n_new_samples

        # Track state across iterations
        self.iteration_data = {}  # Store data for each iteration
        self.best_learner_history = []  # Track best learners across iterations
        self.killed_learners_per_iteration = {}  # Track killed learners PER ITERATION (not permanent)
        self.total_learners = 0  # Track total number of learners

        super().__init__(asyncflow)

        # Setup tasks
        self._setup_tasks()

    def _setup_tasks(self):
        """Setup ROSE tasks using custom Learner approach"""
        
        self.simulation_task = self.simulation_task(as_executable=False)(simulation_task)
        self.training_task = self.training_task(as_executable=False)(training_task)
        self.active_learn_task = self.active_learn_task(as_executable=False)(active_learn_task)

    
    def select_best_learner(self, completed_results: List[LearnerResult], iteration: int) -> Optional[Dict[str, Any]]:
        """Comparator function to select the best model from completed learners"""
        successful_results = [
            result for result in completed_results 
            if result.success and not result.killed
        ]

        if not successful_results:
            logger.info(f"No learners completed successfully in iteration {iteration}")
            return None

        # Find best performing learner
        best_result = max(successful_results, key=lambda x: x.performance_score)

        logger.info(f'+++++++++++++This is the succ. results: {successful_results}')

        # Convert to dict format for compatibility
        best_learner = {
            "learner_id": best_result.learner_id,
            "performance_score": best_result.performance_score,
            "completion_time": best_result.completion_time,
            "model_params": best_result.model_data.get("model_params", {}),
            "iteration": iteration,
            "trained_model": best_result.model_data.get("trained_model")  # Include model if available
        }

        # Add to history for next iteration
        self.best_learner_history.append(best_learner)

        logger.info(f"Best learner: {best_learner['learner_id']} with score {best_learner['performance_score']:.3f}")

        return best_learner

    async def run_parallel_learners_with_dynamic_timeout(self, 
                                                         ml_data: Dict[str, Any], 
                                                         learner_configs: List[Dict],
                                                         iteration: int) -> Tuple[List[LearnerResult], List[int]]:
        """
        Run ALL learners in parallel with DYNAMIC timeout based on first completion
        
        All tasks start in parallel
        asyncio.wait() blocks until exactly one task completes
        Calculate dynamic_timeout = first_completion_time � a
        Apply timeout to remaining tasks using asyncio.wait() with timeout
        Cancel any tasks that don't finish within the timeout
        """
        logger.info(f"Running ALL {len(learner_configs)} learners in parallel for iteration {iteration}...")

        # Create training tasks for ALL learners
        task_futures = []
        learner_id_to_task = {}
        
        for learner_id, config in enumerate(learner_configs):
            task_kwargs = {
                "learner_id": learner_id,
                "iteration": iteration,
                "ml_data": ml_data,
                **config
            }
            task_future = self.training_task(**task_kwargs)
            task_futures.append(task_future)
            learner_id_to_task[id(task_future)] = learner_id

        completed_results = []
        killed_this_iteration = []

        # Wait for FIRST completion only
        done, pending = await asyncio.wait(task_futures, return_when=asyncio.FIRST_COMPLETED)
        
        # Get the first completed task
        first_task = done.pop()
        first_result = await first_task
        first_learner_id = first_result["learner_id"]
        first_completion_time = first_result["completion_time"]

        # Calculate dynamic timeout
        dynamic_timeout = first_completion_time * self.alpha
        logger.info(f"First learner {first_learner_id} completed in {first_completion_time:.1f}s")
        logger.info(f"Dynamic timeout set to {dynamic_timeout:.1f}s (a={self.alpha})")

        # Add first result
        first_learner_result = LearnerResult(
            learner_id=first_result["learner_id"],
            performance_score=first_result["performance_score"],
            completion_time=first_result["completion_time"],
            model_data=first_result,
            success=first_result["success"],
            killed=first_result["killed"]
        )
        completed_results.append(first_learner_result)

        # Apply timeout to remaining tasks
        if pending:
            logger.info(f"Applying {dynamic_timeout:.1f}s timeout to {len(pending)} remaining learners...")
            
            try:
                # Wait for remaining tasks with timeout
                remaining_done, remaining_pending = await asyncio.wait(
                    pending, 
                    timeout=dynamic_timeout,
                    return_when=asyncio.ALL_COMPLETED
                )

                # Process successfully completed tasks
                for task in remaining_done:
                    try:
                        result = await task
                        learner_result = LearnerResult(
                            learner_id=result["learner_id"],
                            performance_score=result["performance_score"],
                            completion_time=result["completion_time"],
                            model_data=result,
                            success=result["success"],
                            killed=result["killed"]
                        )
                        completed_results.append(learner_result)
                    except Exception as e:
                        logger.info(f"Task failed: {e}")
                
                # Cancel and mark killed any tasks that didn't complete
                for task in remaining_pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                    # Find learner_id for this task (approximate - we'll use task position)
                    task_index = task_futures.index(task) if task in task_futures else -1
                    killed_learner_id = task_index if task_index != -1 else len(killed_this_iteration)
                    
                    logger.info(f"Learner {killed_learner_id} KILLED (exceeded {dynamic_timeout:.1f}s)")
                    killed_this_iteration.append(killed_learner_id)
                    
                    # Add killed result
                    killed_result = LearnerResult(
                        learner_id=killed_learner_id,
                        performance_score=0.0,
                        completion_time=first_completion_time + dynamic_timeout,
                        model_data={"kill_reason": "dynamic_timeout_exceeded"},
                        success=False,
                        killed=True
                    )
                    completed_results.append(killed_result)
                    
            except asyncio.TimeoutError:
                # This shouldn't happen with the way we're using wait()
                logger.info("Unexpected timeout in remaining tasks")

        return completed_results, killed_this_iteration
    
    async def teach(self, 
                    initial_simulation_config: Dict,
                    learner_configs: List[Dict]) -> Dict[str, Any]:
        """
        Run the complete multi-iteration synchronization workflow using custom Learner approach
        - Iteration 1: Run simulation task + learners
        - Iteration 2+: Skip simulation, use refined data from previous best learner
        """
        self.total_learners = len(learner_configs)
        
        logger.info(f"Starting workflow: {len(learner_configs)} learners, {self.max_iterations} iterations, a={self.alpha}")
        
        all_iteration_results = []
        current_ml_data = None  # Will hold the dictionary object
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Step 1: Data preparation - simulation ONLY in first iteration
            if iteration == 0:
                simulation_config = {
                    **initial_simulation_config,
                    "iteration": iteration
                }
                current_ml_data = await self.simulation_task(**simulation_config)
                data_source = "initial_simulation"
            else:
                data_source = f"data_x_from_iteration_{iteration}"

            if not current_ml_data:
                logger.info(f"No data available for iteration {iteration + 1}")
                break
            
            logger.info(f"Training data shape: {current_ml_data['x_train'].shape}")
            if current_ml_data.get('x_remain') is not None:
                logger.info(f"Remaining data shape: {current_ml_data['x_remain'].shape}")
            
            # Step 2: Run ALL learners in parallel with DYNAMIC timeout
            completed_results, killed_this_iteration = await self.run_parallel_learners_with_dynamic_timeout(
                current_ml_data, learner_configs, iteration
            )
            
            successful_count = len([r for r in completed_results if r.success and not r.killed])
            if killed_this_iteration:
                logger.info(f"Killed: {killed_this_iteration}")
            
            # Step 3: Wait phase - select best model (comparator)
            best_learner = self.select_best_learner(completed_results, iteration)
            
            if best_learner:
                logger.info(f"Best: Learner {best_learner['learner_id']} (score: {best_learner['performance_score']:.3f})")

            
            # Step 4: Generate Data X for next iteration (if not last iteration)
            data_x_generated = False
            if iteration < self.max_iterations - 1 and best_learner:
                new_ml_data = await self.active_learn_task(
                    selected_learner=best_learner,
                    iteration=iteration,
                    ml_data=current_ml_data,
                    config_path=learner_configs[best_learner['learner_id']].get('config_path', ''),
                    n_new_samples=self.n_new_samples
                )

                # Update current_ml_data for next iteration
                if new_ml_data:
                    current_ml_data = new_ml_data
                    data_x_generated = True
                    logger.info(f"Obtaining data from best model for {iteration + 2}")

            # Prepare iteration results
            successful_learners = [r for r in completed_results if r.success and not r.killed]

            iteration_result = {
                "iteration": iteration + 1,
                "success": len(successful_learners) > 0,
                "total_learners": len(learner_configs),
                "completed_learners": len(successful_learners),
                "killed_this_iteration": killed_this_iteration,
                "kill_count_this_iteration": len(killed_this_iteration),
                "selected_learner": best_learner,
                "data_source": data_source,
                "simulation_run": iteration == 0,
                "data_x_generated": data_x_generated,
                "training_data_info": {
                    "train_shape": current_ml_data["x_train"].shape,
                    "test_shape": current_ml_data["x_test"].shape,
                    "remain_shape": current_ml_data["x_remain"].shape if current_ml_data.get("x_remain") is not None else None,
                } if current_ml_data else None,
                "learner_results": completed_results,
                "iteration_summary": {
                    "total_learners": len(learner_configs),
                    "successful_learners": len(successful_learners),
                    "success_rate": len(successful_learners) / len(learner_configs),
                    "kill_rate": len(killed_this_iteration) / len(learner_configs),
                    "best_performance": best_learner["performance_score"] if best_learner else 0.0
                }
            }

            all_iteration_results.append(iteration_result)
            
            if not iteration_result["success"]:
                logger.info(f"Iteration {iteration + 1} failed - no successful learners")
        
        # Prepare final workflow results
        successful_iterations = [r for r in all_iteration_results if r.get("success", False)]

        # Calculate per-iteration kill statistics
        kill_stats_per_iteration = {}
        for iter_result in all_iteration_results:
            iter_num = iter_result["iteration"]
            kill_stats_per_iteration[iter_num] = {
                "killed_count": iter_result["kill_count_this_iteration"],
                "killed_learners": iter_result["killed_this_iteration"],
                "success_rate": iter_result["iteration_summary"]["success_rate"],
                "kill_rate": iter_result["iteration_summary"]["kill_rate"]
            }
        
        final_result = {
            "workflow_success": len(successful_iterations) > 0,
            "total_iterations": len(all_iteration_results),
            "successful_iterations": len(successful_iterations),
            "iteration_results": all_iteration_results,
            "best_learner_history": self.best_learner_history,
            "final_best_learner": self.best_learner_history[-1] if self.best_learner_history else None,
            "final_ml_data": current_ml_data,  # Include final data state
            "kill_stats_per_iteration": kill_stats_per_iteration,
            "workflow_summary": {
                "completion_rate": len(successful_iterations) / len(all_iteration_results) if all_iteration_results else 0,
                "performance_progression": [bl["performance_score"] for bl in self.best_learner_history],
                "data_flow_iterations": len([r for r in all_iteration_results if r.get("data_x_generated", False)]),
                "average_kill_rate": sum(stats["kill_rate"] for stats in kill_stats_per_iteration.values()) / len(kill_stats_per_iteration) if kill_stats_per_iteration else 0,
                "average_success_rate": sum(stats["success_rate"] for stats in kill_stats_per_iteration.values()) / len(kill_stats_per_iteration) if kill_stats_per_iteration else 0
            }
        }

        return final_result

    async def shutdown(self):
        """Clean shutdown of ROSE components"""
        await self.asyncflow.shutdown()

async def main():
    """Example usage with your actual ML pipeline"""
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    #engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    init_default_logger(logging.DEBUG)
    asyncflow = await WorkflowEngine.create(engine)

    learner = SynchronizedLearner(
        asyncflow,
        alpha=1,  # Kill threshold = 1.5 * first_completion_time (t_min calculated dynamically)
        max_iterations=10,  # Reduced for testing
        n_new_samples=400
    )

    path = os.getcwd()

    # Configure for your actual pipeline - exact paths from your files
    initial_simulation_config = {
        "input_data_dir": f"{os.path.join(path, 'data')}",
        "seed": 42
    }

    # Configure different learners with different ML configs - matching your actual setup
    learner_configs = [
        {  # GPR learner 1
            "learner_name": "GPR-01",
            "config_path": f"{os.path.join(path, 'config', 'gpr.yaml')}",
        },
        {  # BNN learner 
            "learner_name": "BNN-01",
            "config_path": f"{os.path.join(path, 'config', 'bnn.yaml')}",
        },
        {  # MCD learner 
            "learner_name": "MCD-01",
            "config_path": f"{os.path.join(path, 'config', 'mcd.yaml')}",
        }
    ]

    try:
        result = await learner.teach(
            initial_simulation_config=initial_simulation_config,
            learner_configs=learner_configs
        )

        if result["workflow_success"]:
            logger.info(f"Completed {result['successful_iterations']}/{result['total_iterations']} iterations")

            # Show progression through iterations
            for iter_result in result["iteration_results"]:
                iter_num = iter_result["iteration"]
                success_rate = iter_result["iteration_summary"]["success_rate"]
                kill_rate = iter_result["iteration_summary"]["kill_rate"]
                best_perf = iter_result["iteration_summary"]["best_performance"]
                logger.info(f"Iteration {iter_num}: {success_rate:.1%} success, {kill_rate:.1%} killed, best score: {best_perf:.3f}")

            if result["final_best_learner"]:
                best = result["final_best_learner"]
                rmse = best.get('model_params', {}).get('rmse', 'N/A')
                r2 = best.get('model_params', {}).get('R2', 'N/A')
                spearman = best.get('model_params', {}).get('Spearman', 'N/A')
                logger.info(f"Final best learner: {best['learner_id']} (RMSE: {rmse:.4f}), (R2: {r2:.4f}), (Spearman: {spearman:.4f})")

            # Show final data statistics
            if result["final_ml_data"]:
                final_data = result["final_ml_data"]
                logger.info(f"Final data sizes:")
                logger.info(f"Training: {final_data['x_train'].shape}")
                logger.info(f"Test: {final_data['x_test'].shape}")
                if final_data.get('x_remain') is not None:
                    logger.info(f"Remaining: {final_data['x_remain'].shape}")
            
            # Performance progression
            if result["best_learner_history"]:
                scores = [bl["performance_score"] for bl in result["best_learner_history"]]
                learners_best = [ll["learner_id"] for ll in result["best_learner_history"]]
                r2s = [rl["model_params"]["R2"] for rl in result["best_learner_history"]]
                sps = [sl["model_params"]["Spearman"] for sl in result["best_learner_history"]]
                time_perf = [tl["completion_time"] for tl in result["best_learner_history"]]

                logger.info(f"Leaner ID: {' -> '.join([str(s) for s in learners_best])}")
                logger.info(f"Performance progression: {' -> '.join([f'{s:.3f}' for s in scores])}")
                logger.info(f"R2: {' -> '.join([f'{s:.3f}' for s in r2s])}")
                logger.info(f"Spearman: {' -> '.join([f'{s:.3f}' for s in sps])}")
                logger.info(f"Time: {' -> '.join([f'{s:.1f}' for s in time_perf])}")
        else:
            logger.info("Pipeline failed - no successful iterations")
    
    except Exception as e:
        logger.info(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await learner.shutdown()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())