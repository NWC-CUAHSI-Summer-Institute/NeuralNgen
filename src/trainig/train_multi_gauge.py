"""
Multi-gauge training loop for CatchmentLSTM using strictly-proper
scoring-rule losses (FDC/RFL divergence) plus an optional timing-aware
pointwise MSE term, with cluster-proportional gauge batching and
area-weighted aggregation to gauge scale (matching how t-route physically
sums catchment outputs downstream).
"""

from pathlib import Path

import numpy as np
import torch

from src.dataset.multi_gauge_batcher import GaugeHandle, MultiGaugeBatcher
from src.models.catchment_lstm import CatchmentLSTM
from src.losses.scoring_rules import fdc_divergence, rfl_divergence, pointwise_mse


def compute_global_stats(handles):
    """Global dyn_mean/std and stat_mean/std across all gauges' catchments,
    for input normalization."""
    all_dyn = [h._dyn.reshape(-1, h._dyn.shape[-1]) for h in handles.values()]
    all_stat = [h.static for h in handles.values()]
    all_dyn = np.concatenate(all_dyn, axis=0)
    all_stat = np.concatenate(all_stat, axis=0)

    dyn_mean = torch.as_tensor(np.nanmean(all_dyn, axis=0), dtype=torch.float32)
    dyn_std = torch.clamp(torch.as_tensor(np.nanstd(all_dyn, axis=0), dtype=torch.float32), min=1e-6)
    stat_mean = torch.as_tensor(np.nanmean(all_stat, axis=0), dtype=torch.float32)
    stat_std = torch.clamp(torch.as_tensor(np.nanstd(all_stat, axis=0), dtype=torch.float32), min=1e-6)
    return dyn_mean, dyn_std, stat_mean, stat_std


def compute_q_stats(handles):
    """Per-gauge (mean, std) of observed streamflow, for output normalization."""
    q_stats = {}
    for gid, h in handles.items():
        obs = h.q_obs.values
        obs = obs[~np.isnan(obs)]
        mean = float(np.mean(obs)) if len(obs) else 0.0
        std = float(np.std(obs)) if len(obs) else 1.0
        q_stats[gid] = (mean, max(std, 1e-6))
    return q_stats


def train(cfg: dict):
    device = torch.device(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))

    basin_ids = [line.strip() for line in open(cfg["basin_file"]) if line.strip()]
    dynamic_inputs = cfg["dynamic_inputs"]
    static_attributes = cfg["static_attributes"]

    print(f"Loading {len(basin_ids)} gauges...")
    handles = {}
    for gid in basin_ids:
        try:
            handles[gid] = GaugeHandle(
                gid, cfg["gauges_root"], cfg["streamflow_dir"], dynamic_inputs, static_attributes
            )
        except Exception as e:
            print(f"  skipping {gid}: {e}")
    print(f"Loaded {len(handles)} / {len(basin_ids)} gauges.")

    dyn_mean, dyn_std, stat_mean, stat_std = compute_global_stats(handles)
    q_stats = compute_q_stats(handles)
    batcher = MultiGaugeBatcher(handles, cfg["clusters_csv"], exclude_singleton_clusters=True)

    model = CatchmentLSTM(
        dynamic_size=len(dynamic_inputs),
        static_size=len(static_attributes),
        hidden_size=cfg.get("hidden_size", 64),
        num_layers=cfg.get("num_layers", 2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    loss_type = cfg.get("loss_type", "both")       # "fdc", "rfl", "both"
    rfl_weight = cfg.get("rfl_weight", 100.0)       # corrected calibration (was 16.0)
    pointwise_weight = cfg.get("pointwise_weight", 0.0)

    seq_len = cfg.get("seq_len", 336)
    batch_gauges = cfg.get("batch_gauges", 8)
    num_epochs = cfg.get("num_epochs", 50)
    steps_per_epoch = cfg.get("steps_per_epoch", 300)
    rng = np.random.default_rng(cfg.get("seed", 0))

    dyn_mean_d, dyn_std_d = dyn_mean.to(device), dyn_std.to(device)
    stat_mean_d, stat_std_d = stat_mean.to(device), stat_std.to(device)

    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            gauge_ids = batcher.select_batch_gauges(batch_gauges, rng)
            optimizer.zero_grad()
            step_loss = torch.tensor(0.0, device=device)
            n_valid = 0

            for gid in gauge_ids:
                handle = handles[gid]
                max_start = len(handle.time_index) - seq_len
                if max_start <= 0:
                    continue
                start_idx = int(rng.integers(0, max_start))
                dyn, obs = handle.get_window(start_idx, start_idx + seq_len)

                dyn_t = torch.as_tensor(dyn, dtype=torch.float32, device=device)
                dyn_norm = (dyn_t - dyn_mean_d) / dyn_std_d
                static_t = torch.as_tensor(handle.static, dtype=torch.float32, device=device)
                static_norm = (static_t - stat_mean_d) / stat_std_d

                pred_norm, _ = model(dyn_norm, static_norm)  # (n_catchments, seq_len)

                q_mean, q_std = q_stats[gid]
                pred_physical = pred_norm * q_std + q_mean

                areas = torch.as_tensor(handle.areas, dtype=torch.float32, device=device)
                pred_agg = (pred_physical * areas.unsqueeze(1)).sum(dim=0) / areas.sum()

                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

                gauge_loss = torch.tensor(0.0, device=device)
                if loss_type in ("fdc", "both"):
                    gauge_loss = gauge_loss + fdc_divergence(pred_physical, obs_t)
                if loss_type in ("rfl", "both"):
                    gauge_loss = gauge_loss + rfl_weight * rfl_divergence(pred_physical, obs_t)
                if pointwise_weight > 0.0:
                    gauge_loss = gauge_loss + pointwise_weight * pointwise_mse(pred_agg, obs_t)

                step_loss = step_loss + gauge_loss
                n_valid += 1

            if n_valid == 0:
                continue
            step_loss = step_loss / n_valid
            step_loss.backward()
            optimizer.step()
            epoch_loss += step_loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}  loss={epoch_loss/steps_per_epoch:.6f}")

    torch.save({
        "model_state": model.state_dict(),
        "config": {**cfg, "hidden_size": cfg.get("hidden_size", 64), "num_layers": cfg.get("num_layers", 2)},
        "dyn_mean": dyn_mean, "dyn_std": dyn_std,
        "stat_mean": stat_mean, "stat_std": stat_std,
        "q_stats": q_stats,
    }, run_dir / "model_final.pt")
    print(f"Saved checkpoint to {run_dir / 'model_final.pt'}")
