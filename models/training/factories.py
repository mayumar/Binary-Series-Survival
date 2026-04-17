from torch import nn
import torch

from config.schema import LSTMModelConfig, TrainingConfig
from models.training.context import TrainingContext
from models.lstm.LSTM import Survival_LSTM
from models.utils.losses import DiscreteHazardNLL

def build_lstm_model(cfg: LSTMModelConfig, ctx: TrainingContext) -> nn.Module:
    return Survival_LSTM(
        input_size=ctx.input_size,
        num_events=cfg.num_events,
        num_times=ctx.num_time_bins,
        hidden_size1=cfg.hidden_size1,
        hidden_size2=cfg.hidden_size2
    ).to(ctx.device)

def build_loss() -> nn.Module:
    return DiscreteHazardNLL()

def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )