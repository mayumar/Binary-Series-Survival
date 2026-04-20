import torch
import torch.nn as nn

class DiscreteHazardNLL(nn.Module):
    """Negative log-likelihood for a discrete-time hazard model plus a ranking term.

    This module assumes a *single* hazard head of shape ``(B, 1, T)`` (e.g., sigmoid output)
    and computes:

    - Discrete-time hazard negative log-likelihood (event vs. censoring)
    - Pairwise ranking loss (encourages higher cumulative risk for earlier failures)

    :param eps: Numerical stability constant used to clamp hazards to ``[eps, 1-eps]``.
    :type eps: float
    :param sigma: Temperature/smoothing parameter used in the ranking loss.
    :type sigma: float
    """

    def __init__(self, eps: float = 1e-8, sigma: float = 0.2) -> None:
        super().__init__()
        self.eps = float(eps)
        self.sigma = float(sigma)

    def ranking_loss(self, prediction: torch.Tensor, y: torch.Tensor, censor: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss on the cumulative incidence (CDF) over time.

        Penalizes cases where a subject ``i`` who fails earlier than subject ``j`` has
        lower (or not sufficiently higher) estimated cumulative risk at time ``y_i``.

        The loss is computed as::

            exp(-(R_i - R_j) / sigma)

        where ``R`` is the CDF evaluated at time ``y_i``.

        :param prediction: Probability tensor of shape ``(B, E, T)``.
            For this class it is typically ``(B, 1, T)`` (single event),
            but the implementation supports multiple event channels.
        :type prediction: torch.Tensor
        :param y: Time-bin indices of shape ``(B,)``.
        :type y: torch.Tensor
        :param censor: Censoring indicator of shape ``(B,)`` where ``1`` means censored
            and ``0`` means event observed.
        :type censor: torch.Tensor
        :return: Scalar ranking loss.
        :rtype: torch.Tensor
        """
        y = y.long()
        censor = censor.float()

        batch_size, num_events, _ = prediction.shape
        loss_rank = prediction.new_tensor(0.0)

        # Compute the CDF (cumulative risk) over time for all samples
        # cdf = torch.cumsum(prediction, dim=2)  # (B, E, T)

        haz = prediction.clamp(self.eps, 1.0 - self.eps)
        surv = torch.cumprod(1.0 - haz, dim=2)   # (B, E, T)
        cdf = 1.0 - surv                         # (B, E, T)
        # Iterate over each event type to compute event-specific ranking losses
        for k in range(num_events):
            cdf_k = cdf[:, k, :]  # (B, T)

            y_i = y.view(batch_size, 1)
            y_j = y.view(1, batch_size)

            mask_time_order = (y_i < y_j).float()
            mask_uncensored_i = (1.0 - censor).view(batch_size, 1)
            valid_pairs = mask_time_order * mask_uncensored_i

            if valid_pairs.sum() == 0:
                continue

            idx = torch.arange(batch_size, device=cdf_k.device)

            risk_i = cdf_k[idx, y].unsqueeze(1)   # (B, 1)
            risk_j = cdf_k[:, y].T                # (B, B)

            # Compute risk difference: R_i - R_j
            # We want R_i > R_j (earlier failure should imply higher risk)
            diff = risk_i - risk_j

            # Loss = exp(-(R_i - R_j) / sigma)
            rank_loss_matrix = torch.exp(-diff / self.sigma) * valid_pairs
            loss_rank = loss_rank + rank_loss_matrix.sum() / (valid_pairs.sum() + 1e-8)

        # return loss_rank / float(num_events)
        return loss_rank / float(num_events)

    def forward(self, hazard: torch.Tensor, y: torch.Tensor, censor: torch.Tensor) -> torch.Tensor:
        """Compute the discrete-time hazard NLL plus ranking loss.

        :param hazard: Hazard tensor of shape ``(B, 1, T)`` with values in ``(0, 1)``
            (typically sigmoid output).
        :type hazard: torch.Tensor
        :param y: Bin index tensor of shape ``(B,)``.
            - If event observed (``censor=0``): ``y`` must be in ``[0, T-1]``.
            - If censored (``censor=1``): ``y`` can be in ``[0, T]`` where ``T`` means
              "survived all bins".
        :type y: torch.Tensor
        :param censor: Censoring indicator of shape ``(B,)`` (1=censored, 0=event observed).
        :type censor: torch.Tensor
        :return: Scalar loss value.
        :rtype: torch.Tensor
        :raises ValueError: If any ``y < 0`` or if an event sample has ``y >= T``.
        """
        y = y.long()
        c = censor.float()

        # Clamp hazards for numerical stability
        h = hazard[:, 0, :].clamp(self.eps, 1.0 - self.eps)  # (B, T)
        batch_size, T = h.shape

        # Sanity checks
        if torch.any(y < 0):
            raise ValueError(f"Found y < 0: min={int(y.min())}")

        # Ranking term (computed on the provided hazard tensor)
        rank = self.ranking_loss(hazard, y, c)

        # Events cannot have y == T (only censored samples may use y == T)
        if torch.any((c == 0) & (y >= T)):
            bad = y[(c == 0) & (y >= T)]
            raise ValueError(f"Event y out of range (>=T). T={T}, examples={bad[:10].tolist()}")

        log1m_h = torch.log(1.0 - h) # (B, T)
        cumsum_log1m = torch.cumsum(log1m_h, dim=1) # (B, T)

        # prev_surv = sum_{j<y} log(1 - h_j); for y=0 -> 0
        prev_surv = torch.zeros(batch_size, device=h.device, dtype=h.dtype)
        m_prev = (y > 0) & (y <= T)  # y==T is valid here (means up to T-1)
        if m_prev.any():
            prev_surv[m_prev] = cumsum_log1m[m_prev, y[m_prev] - 1]

        # log(h_y) is only defined for events (y in [0, T-1])
        log_hy = torch.zeros(batch_size, device=h.device, dtype=h.dtype)
        m_event = (c == 0)
        if m_event.any():
            log_hy[m_event] = torch.log(h[m_event, y[m_event]])

        ll_event = prev_surv + log_hy

        # Censored: sum_{j<=y} log(1 - h_j)
        # If y == T -> survived all bins => use the last cumulative value (T-1)
        ll_cens = torch.zeros(batch_size, device=h.device, dtype=h.dtype)
        m_cens = (c == 1)
        if m_cens.any():
            y_c = y[m_cens]
            ll_cens[m_cens] = torch.where(
                y_c >= T,
                cumsum_log1m[m_cens, T - 1],
                cumsum_log1m[m_cens, y_c],
            )

        ll = (1.0 - c) * ll_event + c * ll_cens
        return -ll.mean() #+ 0.05 * rank



class DeepHitLoss(nn.Module):
    """DeepHit-style loss for discrete-time competing risks.

    Computes a weighted sum of:
    - negative log-likelihood (event / censoring),
    - pairwise ranking loss over cumulative incidence,
    - calibration loss (MSE on risk at observed time).

    Shapes
    ------
    - ``prediction``: (B, E, T)
    - ``y``: (B,)
    - ``censor``: (B,) where 1=censored, 0=event observed

    :param alpha: Weight for the log-likelihood term.
    :type alpha: float
    :param beta: Weight for the ranking term.
    :type beta: float
    :param gamma: Weight for the calibration term.
    :type gamma: float
    :param sigma: Temperature/smoothing parameter for the ranking term.
    :type sigma: float
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.1, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, prediction: torch.Tensor, y: torch.Tensor, censor: torch.Tensor) -> torch.Tensor:
        """Compute the total DeepHit loss.

        :param prediction: Probability tensor of shape ``(B, E, T)`` (e.g., softmax output).
        :type prediction: torch.Tensor
        :param y: Time-bin indices of shape ``(B,)``.
        :type y: torch.Tensor
        :param censor: Censoring indicator of shape ``(B,)`` (1=censored, 0=event).
        :type censor: torch.Tensor
        :return: Scalar loss value.
        :rtype: torch.Tensor
        """
        # Ensure correct formats
        y = y.long()
        censor = censor.float()
        
        # 1. Log-Likelihood Loss (L1)
        loss_loglike = self.log_likelihood_loss(prediction, y, censor)

        # 2. Ranking Loss (L2)
        loss_ranking = self.ranking_loss(prediction, y, censor)

        # 3. Calibration Loss (L3)
        loss_calibration = self.calibration_loss(prediction, y, censor)

        # print(loss_loglike.item(), self.beta * loss_ranking.item(), self.gamma * loss_calibration.item())

        return self.alpha * loss_loglike + self.beta * loss_ranking + self.gamma * loss_calibration

    def log_likelihood_loss(self, prediction: torch.Tensor, y: torch.Tensor, censor: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood for discrete-time competing risks.

        - If event observed (``censor=0``): maximize probability mass at the observed time bin.
        - If censored (``censor=1``): maximize survival beyond the censoring bin via ``1 - CDF(y)``.

        :param prediction: Probability tensor of shape ``(B, E, T)``.
        :type prediction: torch.Tensor
        :param y: Time-bin indices of shape ``(B,)``.
        :type y: torch.Tensor
        :param censor: Censoring indicator of shape ``(B,)`` (1=censored, 0=event).
        :type censor: torch.Tensor
        :return: Scalar NLL loss.
        :rtype: torch.Tensor
        """
        batch_size, num_events, num_bins = prediction.shape

        # Create a mask to select the correct time bin `y` for each sample
        # Mask shape: (batch, num_bins)
        mask_time = torch.zeros(batch_size, num_bins, device=prediction.device)
        mask_time.scatter_(1, y.unsqueeze(1), 1)

        # Expand the mask to cover events: (batch, num_events, num_bins)
        # NOTE: This is only correct if `y` is shared across events (or if you intentionally
        # sum over events as done below).
        mask_time = mask_time.unsqueeze(1).repeat(1, num_events, 1)

        # A) Event case (uncensored): select the exact probability mass at the observed bin
        # Sum over time and events to obtain the scalar P(T=t, K=k) (as implemented here)
        prob_event = (prediction * mask_time).sum(dim=(1, 2), keepdim=True)  # (B, 1)
        # Avoid log(0)
        prob_event = torch.clamp(prob_event, min=1e-8)
        loss_uncensored = (1.0 - censor) * torch.log(prob_event)

        # B) Censoring case: we need the probability of surviving beyond time t
        # S(t) = 1 - CDF(t)
        # CDF(t) is the cumulative sum of probabilities up to t
        cdf = torch.cumsum(prediction, dim=2)

        # Select the CDF at the censoring time
        cdf_at_censoring = (cdf * mask_time).sum(dim=(1, 2))
        survival_prob = 1.0 - cdf_at_censoring
        survival_prob = torch.clamp(survival_prob, min=1e-8)
        loss_censored = censor * torch.log(survival_prob)

        # Combine terms using the censoring mask
        total_loss = loss_uncensored + 1.0 * loss_censored
        # `censor` is 1 if censored, 0 if event
        # total_loss = (1 - censor) * loss_uncensored + censor * loss_censored

        return -total_loss.mean()

    def ranking_loss(self, prediction: torch.Tensor, y: torch.Tensor, censor: torch.Tensor) -> float:
        """Pairwise ranking loss based on cumulative incidence (CDF).

        For each event type ``k``, considers pairs ``(i, j)`` such that:
        - subject ``i`` is not censored (experienced an event),
        - ``y_i < y_j``,

        and penalizes when ``CDF_i,k(y_i) <= CDF_j,k(y_i)``.

        :param prediction: Probability tensor of shape ``(B, E, T)``.
        :type prediction: torch.Tensor
        :param y: Time-bin indices of shape ``(B,)``.
        :type y: torch.Tensor
        :param censor: Censoring indicator of shape ``(B,)`` (1=censored, 0=event).
        :type censor: torch.Tensor
        :return: Scalar ranking loss.
        :rtype: torch.Tensor
        """
        batch_size, num_events, _ = prediction.shape
        loss_rank = 0.0
                
        # Compute the CDF (cumulative risk) over time for all samples
        cdf = torch.cumsum(prediction, dim=2)  # (batch, events, times)

        # Iterate over each event type to compute event-specific ranking losses
        for k in range(num_events):
            # Select the CDF corresponding only to event k
            cdf_k = cdf[:, k, :]  # (batch, times)

            # Build pairwise comparison matrix (i, j)
            # We want to compare subject i (failed at T_i) with subject j (T_j > T_i)

            # 1. Time ordering matrix: T_i < T_j
            y_i = y.unsqueeze(1)  # (batch, 1)
            y_j = y.unsqueeze(0)  # (1, batch)
            mask_time_order = (y_i < y_j).float()

            # 2. Only consider i if it is NOT censored (i.e., actually experienced the event)
            censor_i = censor.unsqueeze(1)  # (batch, 1)
            mask_uncensored_i = (1 - censor_i).float()

            # Final mask of valid comparable pairs
            valid_pairs = mask_time_order * mask_uncensored_i

            if valid_pairs.sum() == 0:
                continue

            # Obtain the estimated risk at time T_i for both subjects
            # Estimated risk = CDF(T_i)

            # Gather risk for subject i at its own event time T_i
            # cdf_k: (batch, times) -> gather uses indices of shape (batch, 1)
            risk_i = torch.gather(cdf_k, 1, y_i).float()  # (batch, 1)

            # Gather risk for subject j evaluated at time T_i
            # (important: evaluated at T_i, not at T_j)
            y_i_mat = y_i.repeat(1, batch_size)
            risk_j = torch.gather(cdf_k, 1, y_i_mat)  # (batch, batch)

            # Compute risk difference: R_i - R_j
            # We want R_i > R_j (earlier failure should imply higher risk)
            # Loss = exp(-(R_i - R_j) / sigma)
            diff = risk_i.repeat(1, batch_size) - risk_j
            rank_loss_matrix = torch.exp(-diff / self.sigma) * valid_pairs

            loss_rank += rank_loss_matrix.sum() / (valid_pairs.sum() + 1e-8)

        loss = loss_rank / num_events
        return loss.item() if isinstance(loss, torch.Tensor) else loss
    
    def calibration_loss(self, prediction: torch.Tensor, y: torch.Tensor, censor: torch.Tensor) -> float:
        """Calibration loss: MSE between risk at observed time and an indicator.

        Implementation note (from original code):
        ``I2`` is computed as::

            I2 = ((1 - censor) == event_id)

        This only makes sense if ``censor`` is not purely a censor-flag but encodes
        event ids in some way. The method is kept unchanged for compatibility.

        :param prediction: Probability tensor of shape ``(B, E, T)``.
        :type prediction: torch.Tensor
        :param y: Time-bin indices of shape ``(B,)``.
        :type y: torch.Tensor
        :param censor: Censoring indicator (or event encoding, depending on data convention).
        :type censor: torch.Tensor
        :return: Scalar calibration loss.
        :rtype: torch.Tensor
        """
        batch_size, num_events, _ = prediction.shape

        # 1) Cumulative incidence / CDF over time: (B, E, T)
        cdf = torch.cumsum(prediction, dim=2)

        # 2) Risk at observed time y_i: r_{i,e} = CDF_{i,e}(y_i)
        # Gather along time axis
        y_idx = y.view(batch_size, 1, 1).expand(batch_size, num_events, 1) # (B, E, 1)
        r = cdf.gather(dim=2, index=y_idx).squeeze(2) # (B, E)

        # 3) I_2 indicator: I_2[i,e]=1 if subject i had event e (k == e+1), else 0
        # k is 0..E, while event index is 1..E
        event_ids = torch.arange(1, num_events + 1, device=prediction.device).view(1, num_events)   # (1,E)
        I2 = ((1 - censor).view(batch_size, 1) == event_ids).float() # (B, E)

        # 4) MSE calibration loss averaged over batch & events
        loss_cal = ((r - I2) ** 2).mean()
        return loss_cal.item()