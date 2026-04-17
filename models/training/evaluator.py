from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
from typing import List, Union

from config.schema import EvalConfig

@dataclass
class EvalMetrics:
    loss: float
    tpr: float
    fpr: float
    mae_h: float
    lead_times: List[float]
    lead_time_h: float
    lead_time_cont_h: float
    n_events: int
    n_cens: int
    mean_p_event: List[float]
    mean_p_cens: List[float]

class SurvivalEvaluator:
    def __init__(self, loss_fn: nn.Module, cfg: EvalConfig):
        self.loss_fn = loss_fn
        self.cfg = cfg

    def evaluate(self, model: nn.Module, loader: DataLoader, device: Union[str, torch.device]) -> EvalMetrics:
        model.eval()
        eps = 1e-8
        bin_edges = torch.tensor(self.cfg.bin_edges, device=device, dtype=torch.float32)
        bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        total_loss = 0.0
        all_results = []

        with torch.no_grad():
            for x, y, c in loader:
                x, y, c = x.to(device), y.to(device), c.to(device)
                hazards = model(x).clamp(eps, 1.0 - eps)
                total_loss += self.loss_fn(hazards, y, c).item()

                survival = torch.exp(torch.cumsum(torch.log(1.0 - hazards), dim=2))
                risk = 1.0 - survival
                survival_prev = torch.ones_like(survival)
                survival_prev[:, :, 1:] = survival[:, :, :-1]
                pmf0 = (hazards * survival_prev)[:, 0, :]

                crossed = risk[:, 0, :] >= self.cfg.tau_alarm
                has_cross = crossed.any(dim=1)
                k_alarm = torch.where(
                    has_cross,
                    crossed.long().argmax(dim=1).to(torch.float32),
                    torch.tensor(-1.0, device=device, dtype=torch.float32)
                )

                all_results.append({
                    "y": y,
                    "c": c,
                    "t_hat": (pmf0 * bin_mid).sum(dim=1),
                    "alarm_flag": (risk[:, 0, -1] >= self.cfg.tau_severity),
                    "k_alarm": k_alarm,
                    "has_cross": has_cross,
                    "pmf": pmf0
                })
        
        if len(loader) == 0 or len(all_results) == 0:
            return EvalMetrics(float("nan"), 0.0, 0.0, float("nan"), [], 0.0, 0.0, 0, 0, [], [])
        
        res = {k: torch.cat([b[k] for b in all_results]).cpu().numpy() for k in all_results[0]}
        event_mask = res["c"] == 0
        cens_mask = res["c"] == 1
        pmf = res["pmf"]

        mean_p_event = pmf[event_mask].mean(axis=0) if event_mask.any() else np.zeros(pmf.shape[1])
        mean_p_cens = pmf[cens_mask].mean(axis=0) if cens_mask.any() else np.zeros(pmf.shape[1])

        tpr = res["alarm_flag"][event_mask].mean() if event_mask.any() else 0.0
        fpr = res["alarm_flag"][cens_mask].mean() if cens_mask.any() else 0.0

        t_true_mid = np.array(bin_mid.cpu())[res["y"][event_mask]]
        mae = np.abs(res["t_hat"][event_mask] - t_true_mid).mean() if event_mask.any() else np.nan

        event_and_alarm = event_mask & res["has_cross"]
        if any(event_and_alarm):
            t_event_start = np.array(self.cfg.bin_edges)[res["y"][event_and_alarm]]
            t_alarm_start = np.array(self.cfg.bin_edges)[res["k_alarm"][event_and_alarm].astype(int)]
            leads = t_event_start - t_alarm_start
            mean_lead = leads.mean()
        else:
            leads = np.array([])
            mean_lead = 0.0

        mean_lead_cont = (
            (np.array(bin_mid.cpu())[res["y"][event_mask]] - res["t_hat"][event_mask]).mean()
            if any(event_mask) else 0.0
        )

        return EvalMetrics(
            loss=total_loss / len(loader),
            tpr=float(tpr),
            fpr=float(fpr),
            mae_h=float(mae),
            lead_times=leads.tolist(),
            lead_time_h=float(mean_lead),
            lead_time_cont_h=float(mean_lead_cont),
            n_events=int(event_mask.sum()),
            n_cens=int(cens_mask.sum()),
            mean_p_event=mean_p_event.tolist(),
            mean_p_cens=mean_p_cens.tolist()
        )