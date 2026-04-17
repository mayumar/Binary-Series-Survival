from torch.utils.data import DataLoader, Subset
import numpy as np

from models.training.context import TrainingContext
from config.schema import RebalanceConfig

def validate_train_loader(train_loader: DataLoader) -> None:
    if not hasattr(train_loader, "num_time_bins"):
        raise ValueError("The DataLoader does not have the 'num_time_bins' attribute.")
    

def infer_training_context(train_loader: DataLoader, device) -> TrainingContext:
    validate_train_loader(train_loader)

    try:
        sample_x, _, _ = next(iter(train_loader))
    except StopIteration as e:
        raise ValueError("El train_loader está vacío.") from e
    
    return TrainingContext(
        input_size=sample_x.shape[-1],
        seq_len=sample_x.shape[-1],
        num_time_bins=train_loader.num_time_bins,
        device=device
    )

def rebalance_censoring(train_loader: DataLoader, cfg: RebalanceConfig) -> DataLoader:
    if not cfg.enabled:
        return train_loader
    
    dataset = train_loader.dataset
    all_indices = np.arange(len(dataset))
    c_values = np.array([int(dataset[i][2]) for i in all_indices])

    idx_no_cens = all_indices[c_values == 0]
    idx_cens = all_indices[c_values == 1]

    max_cens_allowed = int(cfg.max_censor_ratio * len(idx_no_cens))
    if len(idx_cens) > max_cens_allowed:
        idx_cens = np.random.choice(idx_cens, size=max_cens_allowed, replace=False)

    new_indices = np.concatenate([idx_no_cens, idx_cens])
    np.random.shuffle(new_indices) # Para que no se quede ordenado por bloques (no_cens, cens)

    new_loader = DataLoader(
        Subset(dataset, new_indices.tolist()),
        batch_size=train_loader.batch_size,
        shuffle=True
    )

    if hasattr(train_loader, "num_time_bins"):
        new_loader.num_time_bins = train_loader.num_time_bins

    return new_loader