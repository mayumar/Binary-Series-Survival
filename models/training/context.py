from dataclasses import dataclass
from typing import Union
import torch

@dataclass
class TrainingContext:
    input_size: int
    seq_len: int
    num_time_bins: int
    device: Union[str, torch.device]