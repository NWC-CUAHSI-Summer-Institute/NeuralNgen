"""
Entry point: train_multi / evaluate_multi subcommands.

Usage
-----
    python main.py train_multi --config configs/scoring_rule_multi_rfl.yml --gpu 0
    python main.py evaluate_multi --config configs/scoring_rule_multi_rfl.yml --gpu 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


class Config:
    def __init__(self, data: dict):
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no field '{name}'")

    def __setattr__(self, name, value):
        self._data[name] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        return dict(self._data)


def load_config(config_path: Path) -> Config:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    basin_file = Path(raw["basin_file"])
    raw["basins"] = [l.strip() for l in basin_file.read_text().splitlines() if l.strip()]
    return Config(raw)


def resolve_device(gpu_arg) -> torch.device:
    if gpu_arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(f"WARNING: --gpu {gpu_arg} requested but CUDA is not available - falling back to cpu")
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_arg}")


def run_dir_for(config_path: Path) -> Path:
    return Path("runs") / config_path.stem


def cmd_train_multi(args):
    from src.training.train_multi_gauge import train_multi_gauge

    cfg = load_config(args.config)
    device = resolve_device(args.gpu)
    run_dir = run_dir_for(args.config)

    print(f"[train_multi] config={args.config} loss_type={cfg.loss_type} "
          f"basins={len(cfg.basins)} device={device} run_dir={run_dir}")
    train_multi_gauge(cfg, device, run_dir)


def cmd_evaluate_multi(args):
    from src.dataset.multi_gauge_batcher import MultiGaugeBatcher
    from src.models.catchment_lstm import CatchmentLSTM

    cfg = load_config(args.config)
    device = resolve_device(args.gpu)
    run_dir = run_dir_for(args.config)
    ckpt_path = run_dir / "model_final.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} - run train_multi first")

    print(f"[evaluate_multi] config={args.config} device={device} checkpoint={ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    test_start = pd.Timestamp(cfg.test_start_date)
    test_end = pd.Timestamp(cfg.test_end_date)
    batch_window_hours = cfg.get("batch_window_hours", 168)
    n_dyn = len(cfg.dynamic_inputs)
    n_static = len(cfg.static_attributes)

    dyn_mean, dyn_std = ckpt["dyn_mean"], ckpt["dyn_std"]
    stat_mean, stat_std = ckpt["stat_mean"], ckpt["stat_std"]
    q_stats = ckpt["q_stats"]
    gauge_ids = ckpt["gauge_ids"]

    clusters_df = pd.read_csv(cfg.clusters_file, dtype={"gauge_id": str})
    clusters = dict(zip(clusters_df["gauge_id"], clusters_df["Clusters"]))

    model = CatchmentLSTM(
        dynamic_size=n_dyn, static_size=n_static,
        hidden_size=cfg.get("hidden_size", 64), num_layers=cfg.get("num_layers", 2),
        dropout_p=cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    batcher = MultiGaugeBatcher(
        gauge_ids=gauge_ids, clusters=clusters,
        gauges_root=cfg.gauges_root, streamflow_dir=cfg.streamflow_dir,
        dynamic_inputs=cfg.dynamic_inputs, static_attributes=cfg.static_attributes,
    )

    results = []
    with torch.no_grad():
        for gid in batcher.gauge_ids:
            if gid not in q_stats:
                continue
            handle = batcher.handles[gid]
            q_mean, q_std = q_stats[gid]

            start_idx = handle.date_to_idx(test_start)
            end_idx = handle.date_to_idx(test_end)
            n_windows = max(0, (end_idx - start_idx) // batch_window_hours)
            if n_windows == 0:
                continue

            all_pred, all_obs = [], []
            for w in range(n_windows):
                start_date = handle.time_index[start_idx + w * batch_window_hours]
                dyn, obs = handle.get_window_by_date(start_date, batch_window_hours)
                if dyn is None:
                    continue

                dyn_norm = np.clip((dyn - dyn_mean) / dyn_std, -10, 10)
                stat_norm = np.clip((handle.static - stat_mean) / stat_std, -10, 10)

                x_d = torch.tensor(dyn_norm, dtype=torch.float32, device=device)
                x_s = torch.tensor(stat_norm, dtype=torch.float32, device=device)

                pred = model(x_d, x_s)  # (n_catchments, T_b), normalized
                pred_mm_h = torch.clamp(pred * q_std + q_mean, min=0.0)

                areas_t = torch.tensor(handle.areas, dtype=torch.float32, device=device).unsqueeze(1)
                gauge_agg = (pred_mm_h * areas_t).sum(dim=0) / areas_t.sum()  # (T_b,)

                all_pred.append(gauge_agg.cpu().numpy())
                all_obs.append(obs)

            if not all_pred:
                continue

            pred_arr = np.concatenate(all_pred)
            obs_arr = np.concatenate(all_obs)
            valid = ~np.isnan(obs_arr)
            if valid.sum() < 10:
                continue

            p, o = pred_arr[valid], obs_arr[valid]
            nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
            r = np.corrcoef(o, p)[0, 1] if len(o) > 1 else np.nan
            alpha = p.std() / o.std() if o.std() > 0 else np.nan
            beta = p.mean() / o.mean() if o.mean() != 0 else np.nan
            kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

            results.append(dict(gauge=gid, n_valid=int(valid.sum()), NSE=nse, KGE=kge, r=r,
                                 alpha=alpha, beta=beta))

    results_df = pd.DataFrame(results)
    out_csv = run_dir / "test_metrics.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Evaluated {len(results_df)} gauges. Median NSE={results_df['NSE'].median():.3f}, "
          f"median KGE={results_df['KGE'].median():.3f}")
    print(f"Saved to {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train_multi")
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--gpu", type=int, default=None)
    p_train.set_defaults(func=cmd_train_multi)

    p_eval = sub.add_parser("evaluate_multi")
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--gpu", type=int, default=None)
    p_eval.set_defaults(func=cmd_evaluate_multi)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()