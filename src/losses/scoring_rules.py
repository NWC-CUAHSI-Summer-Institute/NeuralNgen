"""
Strictly-proper scoring-rule losses for distributional streamflow training.

  - fdc_divergence  : energy-distance divergence between predicted and
                       observed flow-duration-curve samples (Eq. 11)
  - rfl_divergence  : energy-distance divergence between predicted and
                       observed rate-of-change distributions (Eq. 12)
  - pointwise_mse   : timing-aware pointwise MSE term added alongside the
                       two divergence terms above, since neither is ever
                       paired to a timestep: fdc/rfl flatten before
                       comparing empirical distributions, so pred(t) is
                       never checked against obs(t). NSE/KGE are
                       timestep-paired metrics, so this term closes that gap.

fdc_divergence and rfl_divergence both estimate the (squared) energy
distance between two empirical distributions:

    ED(P, Q) = 2 E|X - Y| - E|X - X'| - E|Y - Y'|,  X, X' ~ P iid, Y, Y' ~ Q iid

a strictly proper scoring rule (Gneiting & Raftery, 2007; Szekely & Rizzo,
2013): ED(P, Q) >= 0, with equality iff P == Q in distribution.
"""

import torch


def _subsample(x: torch.Tensor, max_n: int = 2000) -> torch.Tensor:
    """Randomly subsample a flattened 1-D tensor to at most max_n elements.

    The energy-distance estimator below is O(n^2); for long simulation
    windows we subsample rather than compute on the full series. Sampling
    is without replacement; tensors already <= max_n pass through unchanged.
    """
    x = x.reshape(-1)
    n = x.shape[0]
    if n <= max_n:
        return x
    idx = torch.randperm(n, device=x.device)[:max_n]
    return x[idx]


def _energy_divergence(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Squared energy-distance divergence between two 1-D samples.

    ED(P, Q) = 2 E|X - Y| - E|X - X'| - E|Y - Y'|

    The two self terms exclude the diagonal (i == j): E|X - X| == 0
    trivially, so including it biases the self-term estimate toward zero
    and understates the true within-sample spread.
    """
    pred = pred.reshape(-1)
    obs = obs.reshape(-1)
    n, m = pred.shape[0], obs.shape[0]
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=pred.device if n else obs.device)

    cross = torch.cdist(pred.unsqueeze(0).unsqueeze(-1), obs.unsqueeze(0).unsqueeze(-1), p=1).squeeze(0)
    cross_term = 2.0 * cross.mean()

    if n > 1:
        pd = torch.cdist(pred.unsqueeze(0).unsqueeze(-1), pred.unsqueeze(0).unsqueeze(-1), p=1).squeeze(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=pred.device)
        pred_self = pd[mask].mean()
    else:
        pred_self = torch.tensor(0.0, device=pred.device)

    if m > 1:
        od = torch.cdist(obs.unsqueeze(0).unsqueeze(-1), obs.unsqueeze(0).unsqueeze(-1), p=1).squeeze(0)
        mask = ~torch.eye(m, dtype=torch.bool, device=obs.device)
        obs_self = od[mask].mean()
    else:
        obs_self = torch.tensor(0.0, device=obs.device)

    return cross_term - pred_self - obs_self


def fdc_divergence(pred: torch.Tensor, obs: torch.Tensor, max_n: int = 2000) -> torch.Tensor:
    """Flow-duration-curve divergence (Eq. 11).

    Compares the empirical distribution of predicted vs. observed flow
    VALUES via the energy distance -- pred(t) is never paired to obs(t).
    NaNs in obs are masked out before comparison.
    """
    obs_flat = obs.reshape(-1)
    pred_flat = pred.reshape(-1)
    mask = ~torch.isnan(obs_flat)
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask] if pred_flat.shape == obs_flat.shape else pred_flat

    pred_s = _subsample(pred_valid, max_n)
    obs_s = _subsample(obs_valid, max_n)
    return _energy_divergence(pred_s, obs_s)


def rfl_divergence(pred: torch.Tensor, obs: torch.Tensor, max_n: int = 2000) -> torch.Tensor:
    """Rate-of-change (flow lag) divergence (Eq. 12).

    Same energy-distance comparison as fdc_divergence, applied to the
    first differences of pred and obs along the time axis instead of the
    raw values. Still only compares the two distributions of rates of
    change -- never pairs d(pred)/dt(t) to d(obs)/dt(t).
    """
    obs_flat = obs.reshape(-1)
    pred_flat = pred.reshape(-1)
    mask = ~torch.isnan(obs_flat)
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask] if pred_flat.shape == obs_flat.shape else pred_flat

    if obs_valid.shape[0] < 2:
        return torch.tensor(0.0, device=pred.device)

    obs_diff = obs_valid[1:] - obs_valid[:-1]
    pred_diff = pred_valid[1:] - pred_valid[:-1]

    pred_s = _subsample(pred_diff, max_n)
    obs_s = _subsample(obs_diff, max_n)
    return _energy_divergence(pred_s, obs_s)


def pointwise_mse(pred_agg: torch.Tensor, q_obs: torch.Tensor) -> torch.Tensor:
    """Timing-aware pointwise MSE term.

    Unlike fdc_divergence/rfl_divergence, this DOES pair pred(t) to obs(t)
    directly -- added to close the gap toward NSE/KGE, which are inherently
    timestep-paired metrics.
    """
    mask = ~torch.isnan(q_obs)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_agg.device)
    return ((pred_agg[mask] - q_obs[mask]) ** 2).mean()
