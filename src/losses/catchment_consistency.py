"""
catchment_consistency_penalty

RECONSTRUCTED, not recovered - same situation as catchment_lstm.py: this
file was never pasted into this conversation. Built to satisfy the exact
call site in train_multi_gauge.py:

    consistency_penalties.append(
        catchment_consistency_penalty(pred_mm_h, areas_t.squeeze(1))
    )
    ...
    consistency_loss = torch.stack(consistency_penalties).mean()

    pred_mm_h : (n_catchments, T_b) - PHYSICAL-space (de-normalized, mm/h,
                already clamped >= 0) predictions for every sub-catchment
                feeding ONE gauge, for one gauge at a time (the training
                loop calls this once per gauge in the batch, only when that
                gauge has more than one catchment).
    areas     : (n_catchments,)      - each catchment's area, matching order.

Returns a 0-d scalar tensor (so `torch.stack(...)` across gauges works).

DESIGN INTENT (this is the actual assumption being made - flag it if it
doesn't match your memory of the original penalty's purpose):

The physical motivation named in train_multi_gauge.py's docstring is that
catchment-level predictions get area-weighted aggregated into the gauge
total the same way NextGen's t-route sums area-weighted catchment
contributions. That aggregation implicitly assumes neighboring/nested
catchments within one gauge behave similarly in specific discharge
(mm/h) - wildly divergent predictions between two catchments of similar
size and proximity would be physically suspicious (though not always
wrong - real catchments do differ). This penalty operationalizes that as
an area-weighted variance: how far each catchment's predicted mm/h
deviates from the AREA-WEIGHTED MEAN prediction across all catchments in
that gauge, at each timestep, weighted again by that catchment's own area
share (so a small catchment behaving oddly is penalized less than a large
one, matching how much each one actually matters to the aggregated total).

This is a *regularizer*, not a hard constraint - it should discourage
implausible spatial noise between catchments without preventing genuinely
different sub-basins (different land cover, slope, etc.) from producing
different discharge, since the static attributes going into the model are
what would justify that difference; consistency_weight (opt-in, default
0.0 in train_multi_gauge.py) controls how strongly it's enforced.
"""

import torch


def catchment_consistency_penalty(pred_mm_h: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """Area-weighted variance of catchment-level predictions within one
    gauge, around their area-weighted mean, averaged over the time window.

    pred_mm_h : (n_catchments, T_b) physical-space (mm/h) predictions.
    areas     : (n_catchments,) catchment areas, any consistent unit.

    Returns a 0-d scalar tensor.
    """
    if pred_mm_h.shape[0] != areas.shape[0]:
        raise ValueError(
            f"catchment_consistency_penalty: pred_mm_h has {pred_mm_h.shape[0]} "
            f"catchments but areas has {areas.shape[0]}"
        )

    weights = areas / areas.sum()                       # (n_catchments,)
    weights = weights.unsqueeze(1)                        # (n_catchments, 1)

    weighted_mean = (pred_mm_h * weights).sum(dim=0, keepdim=True)  # (1, T_b)
    deviations_sq = (pred_mm_h - weighted_mean) ** 2                # (n_catchments, T_b)
    weighted_var_per_t = (deviations_sq * weights).sum(dim=0)       # (T_b,)

    return weighted_var_per_t.mean()