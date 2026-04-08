from typing import List, Optional, OrderedDict, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

@dataclass
class EvalMetrics:
    """
    Container for evaluation metrics computed on a validation or test dataset.

    :param loss: Mean loss over the evaluated dataset.
    :type loss: float
    :param tpr: True Positive Rate (recall). Proportion of real events for which the model triggers an alarm.
    :type tpr: float
    :param fpr: False Positive Rate. Proportion of censored samples for which the model triggers an alarm.
    :type fpr: float
    :param mae_h: Mean Absolute Error of the predicted expected time-to-failure.
    :type mae_h: float
    :param lead_times: List of discrete lead times between the alarm bin and the event bin.
    :type lead_times: List[float]
    :param lead_time_h: Mean discrete lead time (difference between event bin start and alarm bin start).
    :type lead_time_h: float
    :param lead_time_cont_h: Mean continuous lead time computed using the expected failure time prediction.
    :type lead_time_cont_h: float
    :param p_cens: Proportion of censored samples in the dataset.
    :type p_cens: float
    :param p_no_cens: Proportion of non-censored samples (observed events) in the dataset.
    :type p_no_cens: float
    """
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

def train(
    model_for_train: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    N_EPOCH: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: Union[torch.device, str],
    patience: int = 30,
    test_loader: Optional[DataLoader] = None,
    scheduler: Optional[ReduceLROnPlateau] = None,
    tau_severity: float = 0.6,
    tau_alarm: float = 0.6,
    bin_edges: List[float] = [0, 1, 4, 8]
) -> nn.Module:
    """
    Train the survival prediction model.

    The function performs iterative training over multiple epochs, evaluates
    the model on validation and test datasets, and applies early stopping
    based on validation loss.

    :param model_for_train: PyTorch model to be trained.
    :type model_for_train: nn.Module
    :param train_loader: DataLoader providing training batches.
    :type train_loader: DataLoader
    :param valid_loader: DataLoader used for validation during training.
    :type valid_loader: DataLoader
    :param N_EPOCH: Number of training epochs.
    :type N_EPOCH: int
    :param optimizer: Optimizer used for parameter updates.
    :type optimizer: torch.optim.Optimizer
    :param loss_fn: Training loss function.
    :type loss_fn: nn.Module
    :param device: Device used for training ("cpu" or "cuda").
    :type device: Union[torch.device, str]
    :param patience: Number of epochs without improvement before early stopping.
    :type patience: int
    :param test_loader: Optional DataLoader for test data.
    :type test_loader: Optional[DataLoader]
    :param scheduler: Optional learning rate scheduler (e.g., ReduceLROnPlateau).
    :type scheduler: Optional[ReduceLROnPlateau]
    :param tau_severity: Threshold applied to predicted failure probability to determine severity.
    :type tau_severity: float
    :param tau_alarm: Threshold used to trigger alarms based on predicted risk.
    :type tau_alarm: float
    :param bin_edges: Time bin boundaries used to discretize the survival function.
    :type bin_edges: List[float]
    :return: Trained model with the best validation weights restored.
    :rtype: nn.Module
    """
    best_f1_macro = -1.0
    best_state = None
    best_loss = np.inf
    model_for_train.to(device)
    n_stop = 0

    print(f"Starting training loop for {N_EPOCH} epochs...")

    for epoch in range(1, N_EPOCH + 1):
        model_for_train.train()
        train_loss_epoch = 0.0
        batch_count = 0

        for x_train, y_train, c_train in train_loader:
            x_train, y_train, c_train = x_train.to(device), y_train.to(device), c_train.to(device)
            optimizer.zero_grad()
            pred = model_for_train(x_train)
            loss = loss_fn(pred, y_train, c_train)
            
            if torch.isnan(loss):
                print(f"\n[CRITICAL ERROR] Loss is NaN at epoch {epoch}. Stopping training.")

                if best_state is not None:
                    model_for_train.load_state_dict(best_state)
                    
                return model_for_train
            
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            batch_count += 1

        avg_train_loss = train_loss_epoch / batch_count if batch_count > 0 else 0.0

        with torch.no_grad():
            val_metrics = val_test_eval(
                model_for_train, valid_loader, loss_fn, device, bin_edges,
                tau_severity, tau_alarm
            )
            
            perc_pos = (
                np.mean(np.array(val_metrics.lead_times) > 0.0)
                if len(val_metrics.lead_times) > 0
                else 0.0
            )

            test_metrics = val_test_eval(
                model_for_train, test_loader, loss_fn, device, bin_edges,
                tau_severity, tau_alarm
            ) if test_loader is not None else None

        if val_metrics.loss < best_loss:
            best_loss = val_metrics.loss
            best_state = OrderedDict(
                (k, v.detach().cpu().clone())
                for k, v in model_for_train.state_dict().items()
            )
            n_stop = 0
            mark = "(*)"
        else:
            n_stop += 1
            mark = ""
            if n_stop == patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics.tpr)
            else:
                scheduler.step()

        print(
            f"\n[Epoch {epoch:03d}] {mark}"
            f"\n----------------------------------------"
            f"\nVAL SET  | Events: {val_metrics.n_events} | Censored: {val_metrics.n_cens}"
            f"\nTRAIN    | Loss: {avg_train_loss:.4f}"
            f"\nVAL      | Loss: {val_metrics.loss:.4f} | "
            f"TPR: {val_metrics.tpr:.4f} | "
            f"FPR: {val_metrics.fpr:.4f} | "
            f"MAE: {val_metrics.mae_h:.4f}"
            f"\n         | Lead: {val_metrics.lead_time_h:.4f} | "
            f"LeadCont: {val_metrics.lead_time_cont_h:.4f} | "
            f"PositiveLeadRatio: {perc_pos:.2f}"
            f"\n         | Mean P(event): {np.round(val_metrics.mean_p_event, 3)}"
            f"\n         | Mean P(cens):  {np.round(val_metrics.mean_p_cens, 3)}"
        )

        if test_metrics is not None:
            print(
                f"\nTEST SET | Events: {test_metrics.n_events} | Censored: {test_metrics.n_cens}"
                f"\nTEST     | Loss: {test_metrics.loss:.4f} | "
                f"TPR: {test_metrics.tpr:.4f} | "
                f"FPR: {test_metrics.fpr:.4f} | "
                f"MAE: {test_metrics.mae_h:.4f} | "
                f"Lead: {test_metrics.lead_time_h:.4f}"
                f"\n         | Mean P(event): {np.round(test_metrics.mean_p_event, 3)}"
                f"\n         | Mean P(cens):  {np.round(test_metrics.mean_p_cens, 3)}"
            )


    if best_state is not None:
        model_for_train.load_state_dict(best_state)
        print(f"Best model loaded with validation loss: {best_loss:.4f}")
        
    return model_for_train

def val_test_eval(
    model: nn.Module,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    device: Union[torch.device, str],
    bin_edges_hours: List[float],
    tau_severity: float,
    tau_alarm: float
) -> EvalMetrics:
    """
    Evaluate the survival model on a validation or test dataset.

    This function computes survival probabilities from predicted hazards,
    derives alarm signals, and calculates several performance metrics such
    as detection accuracy, time-to-event error, and lead time statistics.

    :param model: Trained PyTorch model that outputs hazard predictions.
    :type model: nn.Module
    :param valid_loader: DataLoader containing evaluation samples.
    :type valid_loader: DataLoader
    :param loss_fn: Loss function used to compute the evaluation loss.
    :type loss_fn: nn.Module
    :param device: Device where tensors are processed ("cpu" or "cuda").
    :type device: Union[torch.device, str]
    :param bin_edges_hours: Time bin boundaries (in hours) used to discretize the survival function.
    :type bin_edges_hours: List[float]
    :param tau_severity: Threshold applied to the predicted failure probability to determine severity.
    :type tau_severity: float
    :param tau_alarm: Threshold used to trigger an alarm based on the predicted risk.
    :type tau_alarm: float
    :return: Evaluation metrics computed over the dataset.
    :rtype: EvalMetrics
    """
    model.eval()
    eps = 1e-8
    
    bin_edges = torch.tensor(bin_edges_hours, device=device, dtype=torch.float32)

    # Midpoints of bins used for expected time regression
    bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:]) 
    
    total_loss = 0.0
    all_results = []

    with torch.no_grad():
        for x_val, y_val, c_val in valid_loader:
            x_val, y_val, c_val = x_val.to(device), y_val.to(device), c_val.to(device)

            hazards = model(x_val) # (batch, events, time_bins)
            hazards = hazards.clamp(eps, 1.0 - eps)
            
            loss = loss_fn(hazards, y_val, c_val)
            total_loss += loss.item()

            # Survival S(t) and cumulative risk F(t)
            S = torch.exp(torch.cumsum(torch.log(1.0 - hazards), dim=2)) 
            F = 1.0 - S
            
            # PMF: p(t) = h(t) * S(t-1)
            S_prev = torch.ones_like(S)
            S_prev[:, :, 1:] = S[:, :, :-1]
            pmf = hazards * S_prev
            pmf0 = pmf[:, 0, :] # Asume E = 0 as main event
            
            y_pred = pmf0.argmax(dim=1)
            t_hat = (pmf0 * bin_mid).sum(dim=1)
            
            # Risk at the end of the prediction horizon
            FH = F[:, 0, -1]
            alarm_flag = (FH >= tau_severity)

            # Alarm time (first crossing of tau_alarm)
            crossed = (F[:, 0, :] >= tau_alarm)
            has_cross = crossed.any(dim=1)

            # Sentinel value (-1) for samples that never cross the threshold
            k_alarm = torch.where(
                has_cross, 
                crossed.long().argmax(dim=1).to(torch.float32), 
                torch.tensor(-1.0, device=device, dtype=torch.float32)
            )

            all_results.append({
                'y': y_val,
                'c': c_val,
                'y_pred': y_pred, 
                't_hat': t_hat,
                'alarm_flag': alarm_flag,
                'k_alarm': k_alarm,
                'has_cross': has_cross,
                'pmf': pmf0,
            })

    # Concatenación eficiente
    res = {k: torch.cat([batch[k] for batch in all_results]).cpu().numpy() for k in all_results[0].keys()}
    event_mask = (res['c'] == 0)
    cens_mask = (res['c'] == 1)

    n_events = event_mask.sum()
    n_cens = cens_mask.sum()

    pmf = res["pmf"]

    mean_p_event = (
        pmf[res["c"] == 0].mean(axis=0)
        if event_mask.any() 
        else np.zeros(pmf.shape[1])
    )
    mean_p_cens = (
        pmf[res["c"] == 1].mean(axis=0)
        if cens_mask.any()
        else np.zeros(pmf.shape[1])
    )

    # Alarm Accuracy
    tpr = res['alarm_flag'][event_mask].mean() if event_mask.any() else 0
    fpr = res['alarm_flag'][cens_mask].mean() if cens_mask.any() else 0

    # MAE
    t_true_mid = np.array(bin_mid.cpu())[res['y'][event_mask]]
    mae = np.abs(res['t_hat'][event_mask] - t_true_mid).mean() if event_mask.any() else np.nan

    # Lead Time (Solo eventos donde saltó la alarma)
    # Evento ocurre en: bin_edges[y]
    # Alarma ocurre en: bin_edges[k_alarm]
    event_and_alarm = event_mask & res['has_cross']
    if any(event_and_alarm):
        t_event_start = np.array(bin_edges_hours)[res['y'][event_and_alarm]]
        t_alarm_start = np.array(bin_edges_hours)[res['k_alarm'][event_and_alarm].astype(int)]
        leads = (t_event_start - t_alarm_start)
        mean_lead = leads.mean()
    else:
        leads = np.array([])
        mean_lead = 0.0

    if any(event_mask):
        lead_cont = np.array(bin_mid.cpu())[res['y'][event_mask]] - res['t_hat'][event_mask]
        mean_lead_cont = lead_cont.mean()
    else:
        mean_lead_cont = 0.0

    metrics = {
        "loss": total_loss / len(valid_loader),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "mae_h": float(mae),
        "lead_times": leads.tolist(),
        "lead_time_h": float(mean_lead),
        "lead_time_cont_h": float(mean_lead_cont),
        "n_events": n_events,
        "n_cens": n_cens,
        "mean_p_event": mean_p_event,
        "mean_p_cens": mean_p_cens
    }

    return EvalMetrics(**metrics)
