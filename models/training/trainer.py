import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Optional

from config.schema import TrainingConfig
from models.training.evaluator import SurvivalEvaluator

class Trainer:
    def __init__(self, cfg: TrainingConfig, evaluator: SurvivalEvaluator):
        self.cfg = cfg
        self.evaluator = evaluator

    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module
    ) -> float:
        model.train()
        total_loss = 0.0
        batch_count = 0

        for x, y, c in loader:
            x, y, c = x.to(self.cfg.device), y.to(self.cfg.device), c.to(self.cfg.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y, c)

            if torch.isnan(loss):
                raise RuntimeError("Loss is NaN during training.")
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        return total_loss / batch_count if batch_count > 0 else 0.0
    
    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        test_loader: Optional[DataLoader]
    ) -> nn.Module:
        best_score = np.inf
        best_loss = np.inf
        best_state = None
        n_stop = 0

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(model, train_loader, optimizer, loss_fn)

            train_metrics = self.evaluator.evaluate(model, train_loader, self.cfg.device)
            val_metrics = self.evaluator.evaluate(model, valid_loader, self.cfg.device)
            score = (
                val_metrics.fpr
                - 0.5 * val_metrics.tpr
                + 0.25 * val_metrics.brier_final
                + 0.1 * val_metrics.loss
            )


            # test_metrics = self.evaluator.evaluate(model, test_loader, self.cfg.device) if test_loader is not None else None

            if score < best_score:
                best_score = score
                best_loss = val_metrics.loss
                best_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
                n_stop = 0
                mark = "(*)"
            else:
                n_stop += 1
                mark = ""
                if n_stop >= self.cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            print(
                f"\n[Epoch {epoch:03d}] {mark}"
                f"\n----------------------------------------"
                f"\nTRAIN SET| Events: {train_metrics.n_events} | Censored: {train_metrics.n_cens}"
                f"\nTRAIN    | Loss(step): {train_loss:.4f}"
                f"\nTRAIN    | Loss(eval): {train_metrics.loss:.4f} | "
                f"TPR: {train_metrics.tpr:.4f} | "
                f"FPR: {train_metrics.fpr:.4f} | "
                f"MAE: {train_metrics.mae_h:.4f}"
                f"\n         | Lead: {train_metrics.lead_time_h:.4f} | "
                f"LeadCont: {train_metrics.lead_time_cont_h:.4f}"
                f"\n         | Mean P(event): {np.round(train_metrics.mean_p_event, 3)}"
                f"\n         | Mean P(cens):  {np.round(train_metrics.mean_p_cens, 3)}"
                f"\n         | RiskFinal(event): {train_metrics.mean_risk_final_event:.4f} | "
                f"RiskFinal(cens): {train_metrics.mean_risk_final_cens:.4f} | "
                f"Gap: {train_metrics.risk_gap:.4f} | "
                f"Brier: {train_metrics.brier_final:.4f}"
                f"\nVAL SET  | Events: {val_metrics.n_events} | Censored: {val_metrics.n_cens}"
                f"\nVAL      | Loss: {val_metrics.loss:.4f} | "
                f"TPR: {val_metrics.tpr:.4f} | "
                f"FPR: {val_metrics.fpr:.4f} | "
                f"MAE: {val_metrics.mae_h:.4f}"
                f"\n         | Lead: {val_metrics.lead_time_h:.4f} | "
                f"LeadCont: {val_metrics.lead_time_cont_h:.4f}"
                f"\n         | Mean P(event): {np.round(val_metrics.mean_p_event, 3)}"
                f"\n         | Mean P(cens):  {np.round(val_metrics.mean_p_cens, 3)}"
                f"\n         | RiskFinal(event): {val_metrics.mean_risk_final_event:.4f} | "
                f"RiskFinal(cens): {val_metrics.mean_risk_final_cens:.4f} | "
                f"Gap: {val_metrics.risk_gap:.4f} | "
                f"Brier: {val_metrics.brier_final:.4f}"
                f"\n         | Score: {score:.4f}"
            )



            # if test_metrics is not None:
            #     print(
            #         f"test_loss={test_metrics.loss:.4f} "
            #         f"test_tpr={test_metrics.tpr:.4f} "
            #         f"test_fpr={test_metrics.fpr:.4f}"
            #     )

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"Best model loaded with validation loss: {best_loss:.4f}")

            if test_loader is not None:
                test_metrics = self.evaluator.evaluate(model, test_loader, self.cfg.device)
                print(
                    f"\n----------------------------------------"
                    f"\nTEST SET | Events: {test_metrics.n_events} | Censored: {test_metrics.n_cens}"
                    f"\nTEST     | Loss: {test_metrics.loss:.4f} | "
                    f"TPR: {test_metrics.tpr:.4f} | "
                    f"FPR: {test_metrics.fpr:.4f} | "
                    f"MAE: {test_metrics.mae_h:.4f}"
                    f"\n         | Lead: {test_metrics.lead_time_h:.4f} | "
                    f"LeadCont: {test_metrics.lead_time_cont_h:.4f} | "
                    # f"PositiveLeadRatio: {test_perc_pos:.2f}"
                    f"\n         | Mean P(event): {np.round(test_metrics.mean_p_event, 3)}"
                    f"\n         | Mean P(cens):  {np.round(test_metrics.mean_p_cens, 3)}"
                )

        return model