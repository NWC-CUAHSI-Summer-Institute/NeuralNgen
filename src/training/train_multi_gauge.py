"""
Multi-gauge training matching the paper's Eq. 3-10, REFINED so the training
objective matches physical deployment: instead of pooling raw per-catchment
predictions directly, each gauge's catchments are AREA-WEIGHTED AGGREGATED
into one combined series (matching how NextGen's t-route physically sums
area-weighted catchment contributions) BEFORE the scoring-rule comparison.
The scoring-rule functions themselves are unchanged -- only what gets fed
into them changes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from NeuralNGEN.src.dataset.multi_gauge_batcher import MultiGaugeBatcher
from NeuralNGEN.src.models.catchment_lstm import CatchmentLSTM
from NeuralNGEN.src.losses.scoring_rules import fdc_divergence, rfl_divergence, pointwise_mse
from NeuralNGEN.src.losses.catchment_consistency import catchment_consistency_penalty


def train_multi_gauge(cfg, device: torch.device, run_dir: Path):
    loss_type = cfg.loss_type
    assert loss_type in ("fdc", "rfl", "both", "mse"), f"unknown loss_type: {loss_type}"
    print(f"=== Multi-gauge training (area-weighted aggregation), loss_type={loss_type}, device={device} ===")

    gauge_ids = cfg.basins
    print(f"Basin list: {len(gauge_ids)} gauges")

    clusters_df = pd.read_csv(cfg.clusters_file, dtype={"gauge_id": str})
    clusters = dict(zip(clusters_df["gauge_id"], clusters_df["Clusters"]))
    gauge_ids = [g for g in gauge_ids if g in clusters]
    print(f"{len(gauge_ids)} gauges have cluster assignments, proceeding with those")

    batcher = MultiGaugeBatcher(
        gauge_ids=gauge_ids, clusters=clusters,
        gauges_root=cfg.gauges_root, streamflow_dir=cfg.streamflow_dir,
        dynamic_inputs=cfg.dynamic_inputs, static_attributes=cfg.static_attributes,
        seed=cfg.get("seed", 42),
    )

    train_start_date = pd.Timestamp(cfg.train_start_date)
    train_end_date = pd.Timestamp(cfg.train_end_date)
    print(f"Train window (calendar dates, applied per-gauge): {train_start_date} to {train_end_date}")

    n_dyn = len(cfg.dynamic_inputs)
    n_static = len(cfg.static_attributes)
    batch_window_hours = cfg.get("batch_window_hours", 168)
    G_b = cfg.get("spatial_length", 8)

    print("Sampling for global dynamic-input normalization stats...")
    sample_dyn = []
    rng = np.random.default_rng(cfg.get("seed", 42))
    n_norm_samples = cfg.get("n_norm_samples", 500)
    attempts = 0
    norm_pbar = tqdm(total=n_norm_samples, desc="Normalization samples", unit="sample")
    while len(sample_dyn) < n_norm_samples and attempts < n_norm_samples * 3:
        attempts += 1
        gid = rng.choice(batcher.gauge_ids)
        handle = batcher.handles[gid]
        start_idx = handle.date_to_idx(train_start_date)
        end_idx = handle.date_to_idx(train_end_date)
        if end_idx - start_idx <= batch_window_hours:
            continue
        start_row = rng.integers(start_idx, end_idx - batch_window_hours)
        start_date = handle.time_index[start_row]
        dyn, _ = handle.get_window_by_date(start_date, batch_window_hours)
        if dyn is None:
            continue
        sample_dyn.append(dyn.reshape(-1, n_dyn))
        norm_pbar.update(1)
    norm_pbar.close()
    sample_dyn = np.concatenate(sample_dyn, axis=0)
    dyn_mean = sample_dyn.mean(axis=0)
    dyn_std = sample_dyn.std(axis=0) + 1e-8
    print(f"Global dynamic stats computed from {len(sample_dyn)} sampled rows")

    all_static = np.concatenate([batcher.handles[g].static for g in batcher.gauge_ids], axis=0)
    stat_mean = all_static.mean(axis=0)
    stat_std = all_static.std(axis=0) + 1e-8

    q_stats = {}
    excluded_gauges = []
    for gid in tqdm(list(batcher.gauge_ids), desc="Computing per-gauge q stats", unit="gauge"):
        handle = batcher.handles[gid]
        start_idx = handle.date_to_idx(train_start_date)
        end_idx = handle.date_to_idx(train_end_date)
        q_train = handle.q_obs.values[start_idx:end_idx]
        n_valid = np.sum(~np.isnan(q_train))
        if n_valid < 100:
            print(f"  Excluding gauge {gid}: only {n_valid} valid train-period observations")
            excluded_gauges.append(gid)
            continue
        q_mean = np.nanmean(q_train)
        q_std = np.nanstd(q_train) + 1e-8
        q_stats[gid] = (q_mean, q_std)

    for gid in excluded_gauges:
        batcher.gauge_ids.remove(gid)
        del batcher.handles[gid]
        for cluster_gauges in batcher.cluster_to_gauges.values():
            if gid in cluster_gauges:
                cluster_gauges.remove(gid)
    print(f"Excluded {len(excluded_gauges)} gauges with insufficient train-period data: {excluded_gauges}")
    print(f"Proceeding with {len(batcher.gauge_ids)} gauges")

    torch.manual_seed(cfg.get("seed", 42))
    model = CatchmentLSTM(
        dynamic_size=n_dyn, static_size=n_static,
        hidden_size=cfg.get("hidden_size", 64), num_layers=cfg.get("num_layers", 2),
        dropout_p=cfg.get("dropout", 0.2),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-3), weight_decay=1e-5)

    steps_per_epoch = cfg.get("steps_per_epoch", 200)
    fdc_weight = cfg.get("fdc_weight", 1.0)
    rfl_weight = cfg.get("rfl_weight", 16.0)
    pointwise_weight = cfg.get("pointwise_weight", 0.0)  # 0.0 = off by default, opt-in
    consistency_weight = cfg.get("consistency_weight", 0.0)  # 0.0 = off by default, opt-in
    require_full_window = loss_type in ("rfl", "both")

    diag_count = 0
    max_diag = 5

    epoch_pbar = tqdm(range(cfg.num_epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        running_loss = running_fdc = running_rfl = 0.0
        n_steps = 0

        step_pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch:02d} steps", unit="step", leave=False)
        for step in step_pbar:
            batch = batcher.get_batch(G_b, batch_window_hours, train_start_date, train_end_date)
            selected = batch["selected_gauges"]

            x_d_list, x_s_list, q_list, valid_gauges, n_catchments_per_gauge = [], [], [], [], []
            for gid in selected:
                dyn = batch["per_gauge_dyn"][gid]
                stat = batch["per_gauge_static"][gid]
                q_window = batch["per_gauge_qobs"][gid]

                q_valid = ~np.isnan(q_window)
                if require_full_window and not q_valid.all():
                    continue
                if q_valid.sum() < 10:
                    continue

                q_mean, q_std = q_stats[gid]
                dyn_norm = np.clip((dyn - dyn_mean) / dyn_std, -10, 10)
                stat_norm = np.clip((stat - stat_mean) / stat_std, -10, 10)
                q_norm = (q_window - q_mean) / q_std

                x_d_list.append(torch.tensor(dyn_norm, dtype=torch.float32, device=device))
                x_s_list.append(torch.tensor(stat_norm, dtype=torch.float32, device=device))
                q_list.append(torch.tensor(q_norm, dtype=torch.float32, device=device))
                valid_gauges.append(gid)
                n_catchments_per_gauge.append(dyn.shape[0])

            if len(valid_gauges) == 0:
                continue

            x_d = torch.cat(x_d_list, dim=0)
            x_s = torch.cat(x_s_list, dim=0)
            q_b = torch.stack(q_list, dim=0)  # (n_valid_gauges, T_b)

            optimizer.zero_grad()
            pred = model(x_d, x_s)  # (total_catchments_in_batch, T_b), normalized space

            # --- area-weighted aggregation: de-normalize each catchment's
            # prediction to physical mm/h (using ITS gauge's own scale),
            # aggregate by area within that gauge, re-normalize the
            # aggregated series (same gauge's scale) for cross-gauge pooling.
            pred_chunks = torch.split(pred, n_catchments_per_gauge, dim=0)
            pred_agg_list = []
            consistency_penalties = []
            for gid, pred_chunk in zip(valid_gauges, pred_chunks):
                q_mean, q_std = q_stats[gid]
                pred_mm_h = pred_chunk * q_std + q_mean
                pred_mm_h = torch.clamp(pred_mm_h, min=0.0)

                areas = batch["per_gauge_areas"][gid]
                areas_t = torch.tensor(areas, dtype=torch.float32, device=device).unsqueeze(1)
                total_area = areas_t.sum()

                gauge_agg_mm_h = (pred_mm_h * areas_t).sum(dim=0) / total_area  # (T_b,)
                gauge_agg_norm = (gauge_agg_mm_h - q_mean) / q_std
                pred_agg_list.append(gauge_agg_norm)

                if consistency_weight > 0 and pred_mm_h.shape[0] > 1:
                    consistency_penalties.append(
                        catchment_consistency_penalty(pred_mm_h, areas_t.squeeze(1))
                    )

            pred_agg = torch.stack(pred_agg_list, dim=0)  # (n_valid_gauges, T_b) -- matches q_b shape
            consistency_loss = (torch.stack(consistency_penalties).mean()
                                 if consistency_penalties else torch.tensor(0.0, device=device))

            fdc_val = rfl_val = None
            if loss_type == "fdc":
                q_b_valid = q_b[~torch.isnan(q_b)]
                if q_b_valid.numel() < 10:
                    continue
                pointwise_loss = pointwise_mse(pred_agg, q_b)
                loss = (fdc_divergence(pred_agg, q_b_valid) + consistency_weight * consistency_loss
                        + pointwise_weight * pointwise_loss)
                fdc_val = loss.item()
            elif loss_type == "rfl":
                pointwise_loss = pointwise_mse(pred_agg, q_b)
                loss = (rfl_divergence(pred_agg, q_b) + consistency_weight * consistency_loss
                        + pointwise_weight * pointwise_loss)
                rfl_val = loss.item()
            elif loss_type == "mse":
                # baseline: plain pointwise MSE only, in the same normalized area-weighted
                # aggregated space as fdc/rfl/both (no signature terms)
                pointwise_loss = pointwise_mse(pred_agg, q_b)
                loss = pointwise_loss + consistency_weight * consistency_loss
            else:
                q_b_valid = q_b[~torch.isnan(q_b)]
                if q_b_valid.numel() < 10:
                    continue
                fdc_loss = fdc_divergence(pred_agg, q_b_valid)
                rfl_loss = rfl_divergence(pred_agg, q_b)
                pointwise_loss = pointwise_mse(pred_agg, q_b)
                loss = (fdc_weight * fdc_loss + rfl_weight * rfl_loss
                        + consistency_weight * consistency_loss + pointwise_weight * pointwise_loss)
                fdc_val, rfl_val = fdc_loss.item(), rfl_loss.item()

            if torch.isnan(loss) or torch.isinf(loss):
                if diag_count < max_diag:
                    diag_count += 1
                    print(f"\n!!! NaN/Inf loss at epoch {epoch} step {step} !!!")
                    print(f"  valid_gauges in this batch: {valid_gauges}")
                    print(f"  pred_agg: shape={pred_agg.shape} min={pred_agg.min().item():.4f} "
                          f"max={pred_agg.max().item():.4f} has_nan={torch.isnan(pred_agg).any().item()}")
                    print(f"  q_b: shape={q_b.shape} min={q_b.min().item():.4f} max={q_b.max().item():.4f} "
                          f"has_nan={torch.isnan(q_b).any().item()}")
                else:
                    print(f"  WARNING: NaN/Inf loss at epoch {epoch} step {step}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            if fdc_val is not None:
                running_fdc += fdc_val
            if rfl_val is not None:
                running_rfl += rfl_val
            n_steps += 1
            step_pbar.set_postfix(loss=f"{loss.item():.5f}", valid_steps=n_steps)
        step_pbar.close()

        avg_loss = running_loss / n_steps if n_steps > 0 else float("nan")
        msg = f"[Epoch {epoch:02d}] avg_loss={avg_loss:.5f}"
        if loss_type == "both" and n_steps > 0:
            msg += f" (fdc={running_fdc / n_steps:.5f}, rfl={running_rfl / n_steps:.5f})"
        msg += f" ({n_steps}/{steps_per_epoch} valid steps)"
        print(msg)
        epoch_pbar.set_postfix(avg_loss=f"{avg_loss:.5f}")

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "dyn_mean": dyn_mean, "dyn_std": dyn_std,
        "stat_mean": stat_mean, "stat_std": stat_std,
        "q_stats": q_stats,
        "gauge_ids": batcher.gauge_ids,
        "config": cfg.as_dict(),
        "aggregation": "area_weighted",  # marks this checkpoint as using the refined training objective
    }, run_dir / "model_final.pt")
    print("Saved to", run_dir / "model_final.pt")