from dataclasses import dataclass, field
from typing import Union, List, Optional
import torch

@dataclass
class TrainingConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    device: Union[str, torch.device] = "cuda"

@dataclass
class EvalConfig:
    tau_severity: float = 0.6
    tau_alarm: float = 0.6
    bin_edges: List[float] = field(default_factory=lambda: [0, 1, 4, 8])
    target_horizon: str = "long"
    default_prediction: Optional[List[float]] = None
    default_atol: float = 0.02

@dataclass
class RebalanceConfig:
    enabled: bool = True
    max_censor_ratio: float = 0.5

@dataclass
class LSTMModelConfig:
    num_events: int = 1
    hidden_size1: int = 64
    hidden_size2: int = 128