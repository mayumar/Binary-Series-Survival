from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.lstm.LSTM import Survival_LSTM
from models.utils.functions import train
from models.utils.losses import DiscreteHazardNLL

def rebalance_censoring(train_loader: DataLoader, max_ratio: float = 0.2):
    """
    Randomly reduce censored samples so that they do not exceed a given
    ratio relative to non-censored samples.

    Samples with ``c = 1`` are considered censored, while ``c = 0`` are
    non-censored. If the number of censored samples is larger than the
    allowed ratio, a random subset is selected.

    :param train_loader: DataLoader containing the training dataset.
    :type train_loader: DataLoader
    :param max_ratio: Maximum allowed ratio of censored samples relative
        to non-censored samples.
    :type max_ratio: float
    :return: A new DataLoader with the rebalanced dataset.
    :rtype: DataLoader
    """

    dataset = train_loader.dataset
    
    all_indices = np.arange(len(dataset))
    
    c_values = np.array([int(dataset[i][2]) for i in all_indices])

    idx_no_cens = all_indices[c_values == 0]
    idx_cens = all_indices[c_values == 1]

    n_no_cens = len(idx_no_cens)
    n_cens = len(idx_cens)

    max_cens_allowed = int(max_ratio * n_no_cens)

    print("\n--- Censoring Rebalancing ---")
    print(f"Non-censored: {n_no_cens}")
    print(f"Original censored: {n_cens}")
    print(f"Allowed censored ({max_ratio*100:.0f}%): {max_cens_allowed}")

    if n_cens > max_cens_allowed:
        idx_cens_sampled = np.random.choice(
            idx_cens, size=max_cens_allowed, replace=False
        )
    else:
        idx_cens_sampled = idx_cens

    new_indices = np.concatenate([idx_no_cens, idx_cens_sampled])
    np.random.shuffle(new_indices)

    print(f"Final training size: {len(new_indices)}")
    print("--------------------------------\n")

    new_dataset = Subset(dataset, new_indices.tolist())

    new_loader = DataLoader(
        new_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True
    )

    if hasattr(train_loader, "num_time_bins"):
        new_loader.num_time_bins = train_loader.num_time_bins

    return new_loader


def train_lstm(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 20,
    num_events: int = 1,
    test_loader: Optional[DataLoader] = None,
    device: torch.device = torch.device('cpu'),
    tau_severity: float = 0.6,
    tau_alarm: float = 0.6,
    bin_edges: List[float] = [0, 1, 4, 8]
) -> nn.Module:
    """
    Train the Survival LSTM model for discrete-time survival prediction.

    This function rebalances censored samples in the training dataset, infers the
    input feature size from the first training batch, initializes the model, and
    trains it using a discrete hazard negative log-likelihood loss.

    The training DataLoader must include a custom attribute ``num_time_bins``,
    which defines the number of discrete time intervals predicted by the model.

    :param train_loader: DataLoader containing the training dataset.
    :type train_loader: DataLoader
    :param valid_loader: DataLoader containing the validation dataset.
    :type valid_loader: DataLoader
    :param epochs: Maximum number of training epochs.
    :type epochs: int
    :param lr: Learning rate used by the optimizer.
    :type lr: float
    :param weight_decay: Weight decay applied by the optimizer.
    :type weight_decay: float
    :param early_stopping_patience: Number of epochs without improvement before early stopping.
    :type early_stopping_patience: int
    :param num_events: Number of event types predicted by the model.
    :type num_events: int
    :param test_loader: Optional DataLoader used to evaluate the model during training.
    :type test_loader: Optional[DataLoader]
    :param device: Device where the model will be trained (e.g., CPU or CUDA).
    :type device: torch.device
    :return: Trained model with the best validation performance.
    :rtype: nn.Module
    :raises ValueError: If ``train_loader`` does not contain the attribute ``num_time_bins``.
    :raises ValueError: If ``train_loader`` is empty.
    """

    train_loader = rebalance_censoring(train_loader, max_ratio=0.5)

    if not hasattr(train_loader, 'num_time_bins'):
        raise ValueError("The DataLoader does not have the 'num_time_bins' attribute.")
        
    num_time_bins = train_loader.num_time_bins
    
    try:
        sample_x, _, _ = next(iter(train_loader))
        input_size = sample_x.shape[-1]
        seq_len = sample_x.shape[1]
    except StopIteration as e:
        raise ValueError("El train_loader está vacío.") from e

    print("--- Model Configuration ---")
    print(f"Input Features: {input_size}")
    print(f"Num Events:     {num_events}")
    print(f"Num Time Bins:  {num_time_bins}")
    print(f"Device:         {device}")
    print("---------------------------")

    model = Survival_LSTM(
       input_size=input_size,
       num_events=num_events,
       num_times=num_time_bins, 
       hidden_size1=64,
       hidden_size2=128
    ).to(device)

    # Usamos sigma=1.0 para suavizar gradientes (evita explosiones numéricas)
    # Alpha=0.5 da igual peso a la verosimilitud (qué bin es) y al ranking (orden)
    # loss_fn = DeepHitLoss(alpha=1, beta=0.5, gamma=0.1, sigma=0.1)
    loss_fn = DiscreteHazardNLL()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_model = train(
        model_for_train=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        N_EPOCH=epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        patience=early_stopping_patience,
        device=device,
        tau_severity=tau_severity,
        tau_alarm=tau_alarm,
        bin_edges=bin_edges
    )
    
    return best_model