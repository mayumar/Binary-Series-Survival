import argparse
import os
import json

import torch
from torch.utils.data import DataLoader

from data_loading.data_loader import load_data 
from data_preprocessing.preprocess import preprocess_data
from models.lstm import trainer_lstm

import collections

def load_config(config_path: str) -> dict:
    """
    Load the training configuration from a JSON file.

    :param config_path: Path to the JSON configuration file.
    :type config_path: str
    :return: Parsed configuration dictionary.
    :rtype: dict
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def calculate_horizon(mtbf_f: float, max_h: int) -> int:
    """
    Compute the discrete prediction horizon from the MTBF value.

    The function ensures a minimum horizon of 1 and applies a
    conservative rounding strategy before limiting the result
    to the maximum allowed horizon.

    :param mtbf_f: Mean Time Between Failures value (may be float).
    :type mtbf_f: float
    :param max_h: Maximum allowed horizon.
    :type max_h: int
    :return: Final bounded horizon.
    :rtype: int
    """
    if mtbf_f < 1.0:
        mtbf = 1
        mtbf = int(mtbf)
    elif mtbf_f > 10.0:
        mtbf = int(mtbf_f)
    else:
        mtbf = int(mtbf_f) + 1

    return min(max_h, mtbf)

def debug_data_distribution(loader: DataLoader, num_bins: int):
    """
    Print diagnostic information about censoring ratio and
    time-bin distribution in a survival DataLoader.

    The loader is expected to return batches in the form (x, y, c),
    where:
        - y is the discrete time-bin index.
        - c is the censoring indicator (1 = censored, 0 = event).

    :param loader: DataLoader containing survival batches.
    :type loader: torch.utils.data.DataLoader
    :param num_bins: Number of discrete time bins.
    :type num_bins: int
    :return: None
    :rtype: None
    """
    print("\n--- STARTING DATA DIAGNOSTICS ---")
    all_y = []
    all_not_c_y = []
    all_c = []
    
    for _, y, c in loader:
        mask = (c == 0)
        not_censored_y = y[mask]
        all_y.extend(y.numpy())
        all_not_c_y.extend(not_censored_y.numpy())
        all_c.extend(c.numpy())
        
    total = len(all_y)
    not_c_total = len(all_not_c_y)
    
    # Check censured data
    censored_count = sum(all_c)
    events_count = total - censored_count
    print(f"Total samples: {total}")
    print(f"Observed events (failures): {events_count} ({events_count/total:.2%})")
    print(f"Censored samples: {censored_count} ({censored_count/total:.2%})")
    
    # Check Bin Distribution
    counter = collections.Counter(all_not_c_y)
    if not_c_total != 0:
        print("\nTime-bin distribution (non-censored only):")
        print("Bin | Count | % of non-censored")
        print("-" * 35)
        print("-" * 30)
        for i in range(num_bins):
            count = counter.get(i, 0)
            print(f"  {i} | {count:8d} | {count/not_c_total:.2%}")
    else:
        print("No non-censored samples available.")
        
    # Alerta visual
    if events_count < 100:
        print("\n[CRITICAL WARNING] Very few failure events detected.")
    
    if counter.get(num_bins-1, 0) / total > 0.90:
        print("\n[CRITICAL WARNING] More than 90% of samples are in the last bin.")

    print("--------------------------------------\n")


def execute_training(fail_to_pred: str, config: dict, device_str: str = "cuda"):
    """
    Execute the full survival training pipeline for a given target failure.

    The procedure includes:
        1. Device selection.
        2. Horizon calculation from MTBF.
        3. Data loading and validation checks.
        4. Preprocessing and DataLoader creation.
        5. Model training.
        6. Model persistence.

    :param fail_to_pred: Name of the failure mode (target column).
    :type fail_to_pred: str
    :param config: Configuration dictionary containing paths and parameters.
    :type config: dict
    :param device_str: Device preference ("cuda" or "cpu").
    :type device_str: str
    :return: None
    :rtype: None
    :raises ValueError: If no data is found or validation split is empty.
    """
    
    # 1. Device selection
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Starting training for target: {fail_to_pred}")
    
    # 2. Horizon calculation from MTBF
    with open(config['paths']['mtbfs'], "r") as f:
        mtbfs = json.load(f)

    horizon = calculate_horizon(mtbfs[fail_to_pred], config['parameters']['max_horizon'])

    print(f"Horizon -> {horizon}")

    # 3. Data loading and validation checks
    print("Loading raw data...")
    train_data, val_data, test_data = load_data(
        config['paths']['input_data'],
        fail_to_pred,
        horizon_h=horizon,
        # start_idx=config["parameters"]["start_idx"]
    ) # Return -> (x_train, y_train, c_train), (x_val, y_val, c_val), (x_test, y_test, c_test)

    print("Data loaded successfully")

    n_train = len(train_data[0])
    n_val   = len(val_data[0])
    n_test  = len(test_data[0])

    print(f"\n--- DATA STATUS FOR '{fail_to_pred}' ---")
    print(f"Runs in Train: {n_train}")
    print(f"Runs in Val:   {n_val}")
    print(f"Runs in Test:  {n_test}")
    
    if n_train == 0 and n_val == 0 and n_test == 0:
        raise ValueError(f"CRITICAL: No data found for '{fail_to_pred}'. \n"
                         f"Check the column name in the dataset.")
    
    if n_val == 0:
        raise ValueError(f"CRITICAL: Insufficient data to create validation set. \n"
                         f"Only {n_train} samples available.")

    # 4. Preprocessing and DataLoader creation
    print("Preprocessing and generating embeddings + time bins...")
    create_event2vec_folders(fail_to_pred, config)
    train_loader, val_loader, test_loader = preprocess_data(train_data, val_data, test_data, fail_to_pred, config)

    # Depuracion
    debug_data_distribution(train_loader, train_loader.num_time_bins)

    # 5. Model training
    # num_events=1 porque estamos entrenando un modelo específico para 'fail_to_pred'
    # Si quisieras predecir 2 fallos a la vez, cargarías ambos en 'y' y pondrías num_events=2
    print("Starting model training...")
    
    model = trainer_lstm.train_lstm(
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader, 
        device=device,
        epochs=config['parameters']['epochs'],
        early_stopping_patience=config['parameters']['early_stopping_patience'],
        lr=config['parameters']['learning_rate'],
        num_events=1,
        tau_severity=config['parameters']['tau_severity'],
        tau_alarm=config['parameters']['tau_alarm'],
        bin_edges=config['bin_edges']
    )

    print("Training completed.")
    
    # 6. Model persistence
    torch.save(model.state_dict(), f"saved_models/survival_model_{fail_to_pred}.pth")
    print(f"Model saved as survival_model_{fail_to_pred}.pth")

def create_event2vec_folders(fail_to_pred: str, config: dict):
    event2vec_dir = config["paths"]["event2vec_dir"]
    fail_path = os.path.join(event2vec_dir, fail_to_pred)
    os.makedirs(fail_path, exist_ok=True)

if __name__ == "__main__":
    """
    Command-line entry point for survival model training.

    This script allows training a model trainign for survival analysis
    for a specific target failure mode.

    Command-line arguments:
        --target : Target column name.
        --config : Path to configuration file.
        --device : Execution device (cuda/cpu).
    """
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Model training for survival analysis")
    parser.add_argument("--target", type=str, required=True, help="Target variable (failure mode) to predict")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    os.makedirs("saved_models", exist_ok=True)

    config = load_config(args.config)
    
    execute_training(args.target, config, device_str=args.device)