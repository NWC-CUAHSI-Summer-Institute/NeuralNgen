"""
Evaluate a trained CatchmentLSTM checkpoint across many gauges (standalone,
NOT routed through NextGen/t-route -- see evaluate_ngen_results.py for the
real-routing evaluation).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset.multi_gauge_batcher import GaugeHandle
from src.models.catchment_lstm import CatchmentLSTM

CHUNK_SIZE = 2000


def compute_nse(pred, obs):
    pred, obs = np.asarray(pred, dtype=float), np.asarray(obs, dtype=float)
    mask = ~(np.isnan(pred) | np.isnan(obs))
    pred, obs = pred[mask], obs[mask]
    if len(obs) < 2:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    return np.nan if denom == 0 else 1.0 - np.sum((obs - pred) ** 2) / denom


def compute_kge(pred, obs):
    pred, obs = np.asarray(pred, dtype=float), np.asarray(obs, dtype=float)
    mask = ~(np.isnan(pred) | np.isnan(obs))
    pred, obs = pred[mask], obs[mask]
    if len(obs) < 2 or np.std(obs) == 0 or np.std(pred) == 0 or np.mean(obs) == 0:
        return np.nan
    r = np.corrcoef(pred, obs)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def evaluate(cfg: dict):
    device = torch.device(cfg["device"])
    run_dir = Path(cfg["run_dir"])
    ckpt = torch.load(run_dir / "model_final.pt", map_location=device, weights_only=False)

    train_cfg = ckpt["config"]
    dynamic_inputs = train_cfg["dynamic_inputs"]
    static_attributes = train_cfg["static_attributes"]

    model = CatchmentLSTM(
        dynamic_size=len(dynamic_inputs),
        static_size=len(static_attributes),
        hidden_size=train_cfg.get("hidden_size", 64),
        num_layers=train_cfg.get("num_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dyn_mean, dyn_std = ckpt["dyn_mean"].to(device), ckpt["dyn_std"].to(device)
    stat_mean, stat_std = ckpt["stat_mean"].to(device), ckpt["stat_std"].to(device)

    test_start = pd.Timestamp(cfg["test_start_date"])
    test_end = pd.Timestamp(cfg["test_end_date"])

    basin_ids = [line.strip() for line in open(cfg["basin_file"]) if line.strip()]

    results = []
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for gid in basin_ids:
        try:
            handle = GaugeHandle(gid, cfg["gauges_root"], cfg["streamflow_dir"], dynamic_inputs, static_attributes)
        except Exception as e:
            results.append({"gauge_id": gid, "NSE": np.nan, "KGE": np.nan, "error": str(e)})
            continue

        test_mask = (handle.time_index >= test_start) & (handle.time_index < test_end)
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0:
            results.append({"gauge_id": gid, "NSE": np.nan, "KGE": np.nan, "error": "no test period"})
            continue

        dyn, _ = handle.get_window(int(test_idx.min()), int(test_idx.max()) + 1)
        dyn_t = torch.as_tensor(dyn, dtype=torch.float32, device=device)
        dyn_norm = (dyn_t - dyn_mean) / dyn_std
        static_t = torch.as_tensor(handle.static, dtype=torch.float32, device=device)
        static_norm = (static_t - stat_mean) / stat_std

        preds_chunks = []
        with torch.no_grad():
            for start in range(0, dyn_norm.shape[1], CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, dyn_norm.shape[1])
                pred_chunk, _ = model(dyn_norm[:, start:end, :], static_norm)
                preds_chunks.append(pred_chunk.cpu().numpy())
        pred_norm = np.concatenate(preds_chunks, axis=1)

        q_mean, q_std = ckpt["q_stats"][gid]
        pred_physical = pred_norm * q_std + q_mean

        areas = handle.areas
        pred_aggregate = (pred_physical * areas[:, None]).sum(axis=0) / areas.sum()
        obs_test = handle.q_obs.values[test_idx]

        nse = compute_nse(pred_aggregate, obs_test)
        kge = compute_kge(pred_aggregate, obs_test)
        results.append({"gauge_id": gid, "NSE": nse, "KGE": kge, "error": None})

        out_df = pd.DataFrame({
            "date": handle.time_index[test_idx],
            "pred": pred_aggregate,
            "obs": obs_test,
        })
        out_df.to_csv(pred_dir / f"{gid}_predictions.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv(run_dir / "test_metrics_multi.csv", index=False)

    valid = df[df["NSE"].notna()]
    print(f"\nn={len(valid)} / {len(df)} gauges evaluated")
    if len(valid) > 0:
        print(f"NSE median={valid['NSE'].median():.4f}, KGE median={valid['KGE'].median():.4f}")
        print(valid[["NSE", "KGE"]].describe())
