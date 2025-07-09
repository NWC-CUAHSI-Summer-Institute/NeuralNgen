# src/neuralngen/validate/epoch_validation.py

import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from neuralngen.dataset.hourlycamelsus import HourlyCamelsDataset
from neuralngen.utils.performance_measures import nse
from neuralngen.dataset.normalization import runoff_mmhr_to_cfs, inverse_scale

def validate_epoch(model, cfg, device="cpu", run_dir=None):
    model.eval()

    # Load validation dataset with scaling
    dataset = HourlyCamelsDataset(
        cfg,
        is_train=False,
        period="validation",
        run_dir=run_dir,
        do_load_scalers=True
    )

    # Index samples by basin
    basin_to_indices = defaultdict(list)
    for i, (basin_id, start_idx) in enumerate(dataset.samples):
        basin_to_indices[basin_id].append(i)

    nse_per_basin = {}
    seq_len = cfg.sequence_length

    for basin in dataset.basins:
        indices = basin_to_indices[basin]
        if not indices:
            continue

        # Gather batches
        y_true_batches = []
        y_pred_batches = []
        batch_size_val = 128
        for i in range(0, len(indices), batch_size_val):
            batch_idxs = indices[i:i+batch_size_val]
            x_d = torch.stack([dataset[idx]["x_d"] for idx in batch_idxs]).to(device)
            x_s = torch.stack([dataset[idx]["x_s"] for idx in batch_idxs]).to(device)
            y   = torch.stack([dataset[idx]["y"]   for idx in batch_idxs]).to(device)

            with torch.no_grad():
                out = model(x_d, x_s)
                y_hat = out["y_hat"]

            # remove warm-up
            y_hat = y_hat[:, seq_len:, :].cpu().numpy()
            y_true = y[:, seq_len:, :].cpu().numpy()

            y_pred_batches.append(y_hat)
            y_true_batches.append(y_true)

        # Concatenate all
        y_true_all = np.concatenate(y_true_batches, axis=0)
        y_pred_all = np.concatenate(y_pred_batches, axis=0)

        # Flatten per-variable
        var_names = dataset.cfg.target_variables
        n_vars = len(var_names)
        y_true_flat = y_true_all.reshape(-1, n_vars)
        y_pred_flat = y_pred_all.reshape(-1, n_vars)

        # Inverse-scale to mm/hr
        df_true = pd.DataFrame(y_true_flat, columns=var_names)
        df_pred = pd.DataFrame(y_pred_flat, columns=var_names)
        y_true_phys_df = inverse_scale(df_true, dataset.target_scalers)
        y_pred_phys_df = inverse_scale(df_pred, dataset.target_scalers)

        if n_vars == 1:
            y_true_phys = y_true_phys_df[var_names[0]].values
            y_pred_phys = y_pred_phys_df[var_names[0]].values
        else:
            y_true_phys = y_true_phys_df.values
            y_pred_phys = y_pred_phys_df.values

        # Convert to cfs
        area = dataset.basin_info[basin]["basin_area_km2"]
        y_true_cfs = runoff_mmhr_to_cfs(y_true_phys, area)
        y_pred_cfs = runoff_mmhr_to_cfs(y_pred_phys, area)

        # Summary stats
        true_mean_mm = np.mean(y_true_phys)
        true_std_mm  = np.std(y_true_phys)
        true_min_mm  = np.min(y_true_phys)
        true_max_mm  = np.max(y_true_phys)

        pred_mean_mm = np.mean(y_pred_phys)
        pred_std_mm  = np.std(y_pred_phys)
        pred_min_mm  = np.min(y_pred_phys)
        pred_max_mm  = np.max(y_pred_phys)

        true_mean_cf = np.mean(y_true_cfs)
        true_std_cf  = np.std(y_true_cfs)
        true_min_cf  = np.min(y_true_cfs)
        true_max_cf  = np.max(y_true_cfs)

        pred_mean_cf = np.mean(y_pred_cfs)
        pred_std_cf  = np.std(y_pred_cfs)
        pred_min_cf  = np.min(y_pred_cfs)
        pred_max_cf  = np.max(y_pred_cfs)

        # Compute NSE
        nse_val = nse(y_true_cfs, y_pred_cfs)
        nse_per_basin[basin] = nse_val

        # Detailed print
        print(f"\nBasin {basin} statistics:")
        print(f"  NSE (cfs):      {nse_val:.4f}")
        print(f"  True mean:      {true_mean_mm:.6f} mm/hr ± {true_std_mm:.6f}")
        print(f"  True min/max:   [{true_min_mm:.6f}, {true_max_mm:.6f}] mm/hr")
        print(f"  Pred mean:      {pred_mean_mm:.6f} mm/hr ± {pred_std_mm:.6f}")
        print(f"  Pred min/max:   [{pred_min_mm:.6f}, {pred_max_mm:.6f}] mm/hr")
        print(f"  True mean:      {true_mean_cf:.3f} cfs ± {true_std_cf:.3f}")
        print(f"  True min/max:   [{true_min_cf:.3f}, {true_max_cf:.3f}] cfs")
        print(f"  Pred mean:      {pred_mean_cf:.3f} cfs ± {pred_std_cf:.3f}")
        print(f"  Pred min/max:   [{pred_min_cf:.3f}, {pred_max_cf:.3f}] cfs")
        print(f"  N samples:      {y_true_phys.size}")

    # Aggregate
    vals = np.array(list(nse_per_basin.values()))
    vals = vals[~np.isnan(vals)]
    if vals.size > 0:
        print("\n======= Validation NSE Statistics =======")
        print(f"Mean NSE:   {vals.mean():.4f}")
        print(f"Median NSE: {np.median(vals):.4f}")
        print(f"Min NSE:    {vals.min():.4f}")
        print(f"Max NSE:    {vals.max():.4f}")
    else:
        print("No valid NSE values computed.")

    return nse_per_basin