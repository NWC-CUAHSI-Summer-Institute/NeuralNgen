# src/neuralngen/utils/plot_batches.py

import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuralngen.models.ngenlstm import NgenLSTM
from neuralngen.dataset.hourlycamelsus import HourlyCamelsDataset
from neuralngen.dataset.collate import custom_collate
from neuralngen.utils import Config
from neuralngen.utils.distance import compute_distance_matrix
import re

def create_run_dir(cfg):
    run_dir = Path(cfg.output_dir) / f"{cfg.experiment_name}_plotsample"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created run directory: {run_dir}")
    return run_dir

def sanitize_filename(name):
    """ Replace problematic characters in strings to make safe filenames. """
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

def get_raw_target_for_sample(dataset, basin, start_idx):
    """
    Extracts the pre-scaled (physical units) target data for the same window as the batch sample.
    """
    end_idx = start_idx + dataset.window_size
    # dataset.targets_raw[basin] should be a DataFrame
    return dataset.targets_raw[basin].iloc[start_idx:end_idx].values  # [window_size, n_targets]

def plot_batch_with_raw(batch, raw_targets_batch, cfg, batch_idx, run_dir, scaler_dict):
    """
    Plots raw and scaled target variables for one batch, all in one figure.
    Also prints out a direct comparison for the first site and first variable.
    """
    y = batch["y"].cpu().numpy()       # [B, T, Dout]
    num_sites, T, Dout = y.shape
    target_var_names = cfg.target_variables

    time_axis = np.arange(T)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # === PRINT COMPARISON ===
    print("\n[SCALING COMPARISON PRINT] First site, first variable, first 10 time steps:")
    site = 0
    var_idx = 0
    var_name = target_var_names[var_idx]
    mean = scaler_dict[var_name]["mean"]
    std = scaler_dict[var_name]["std"]

    print(f"Variable: {var_name}")
    print("Raw values:    ", np.round(raw_targets_batch[site, :10, var_idx], 5))
    print("Scaled values: ", np.round(y[site, :10, var_idx], 5))
    print("Scaler mean:   ", mean)
    print("Scaler std:    ", std)
    print("Manual z-score:", np.round((raw_targets_batch[site, :10, var_idx] - mean) / std, 5))
    print("-" * 50)

    fig, axs = plt.subplots(Dout, 2, figsize=(14, Dout * 3), sharex=True)
    if Dout == 1:
        axs = np.array([axs])  # ensure 2D

    for i_var, var_name in enumerate(target_var_names):
        # Left: Pre-scaled (physical units)
        for i_site in range(num_sites):
            axs[i_var, 0].plot(time_axis, raw_targets_batch[i_site, :, i_var], color="lightgrey", linewidth=0.5, alpha=0.7)
        mean_raw = raw_targets_batch[:, :, i_var].mean(axis=0)
        axs[i_var, 0].plot(time_axis, mean_raw, color="green", linewidth=2, label="Mean (raw)")
        axs[i_var, 0].set_title(f"{var_name} (Original units)")
        axs[i_var, 0].set_ylabel(var_name)
        axs[i_var, 0].legend()

        # Right: Scaled (model input)
        for i_site in range(num_sites):
            axs[i_var, 1].plot(time_axis, y[i_site, :, i_var], color="lightgrey", linewidth=0.5, alpha=0.7)
        mean_scaled = y[:, :, i_var].mean(axis=0)
        axs[i_var, 1].plot(time_axis, mean_scaled, color="blue", linewidth=2, label="Mean (scaled)")
        axs[i_var, 1].set_title(f"{var_name} (Scaled for model)")
        axs[i_var, 1].set_ylabel(var_name)
        axs[i_var, 1].legend()

    for ax in axs[-1, :]:
        ax.set_xlabel("Time step")

    plt.suptitle(f"Batch {batch_idx} - Raw vs Scaled Targets", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = plot_dir / f"batch_{batch_idx}_raw_vs_scaled_targets.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved raw vs scaled target plot to {fig_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=2,
        help="How many batches to plot.",
    )
    args = parser.parse_args()

    # Load config
    cfg = Config(Path(args.config))

    # Create run dir
    run_dir = create_run_dir(cfg)

    # Create dataset (scalers will be computed/loaded)
    dataset = HourlyCamelsDataset(
        cfg,
        is_train=True,
        period="train",
        run_dir=run_dir,
        do_load_scalers=True,
    )

    if len(dataset.all_basins_with_samples) == 0:
        print("[ERROR] No basins have samples. Exiting.")
        return

    # --- Ensure raw (pre-scaled) targets are accessible ---
    if not hasattr(dataset, "targets_raw"):
        print("[ERROR] Dataset is missing .targets_raw attribute (pre-scaled target values).")
        print("Add: self.targets_raw[basin] = df[cfg.target_variables] in your dataset __init__.")
        return

    print(f"Number of basins with samples: {len(dataset.all_basins_with_samples)}")
    print(f"Total windows in dataset: {dataset.total_windows}")

    # Grab scaler dictionary for printing
    scaler_dict = dataset.target_scalers

    for batch_idx in range(args.num_batches):
        # Pick basins for this batch
        basins_in_batch = random.sample(
            dataset.all_basins_with_samples,
            k=min(cfg.batch_sites, len(dataset.all_basins_with_samples)),
        )
        
        batch_samples = []
        raw_targets_batch = []
        for basin in basins_in_batch:
            # Randomly select a window for this basin
            start_idx = random.choice(dataset.samples_by_basin[basin])
            sample = dataset._load_window(basin, start_idx)
            batch_samples.append(sample)
            # Also get the original, pre-scaled target values
            raw_targets_batch.append(get_raw_target_for_sample(dataset, basin, start_idx))

        # Collate as usual
        batch = custom_collate(batch_samples)
        raw_targets_batch = np.stack(raw_targets_batch)   # [B, T, Dout]

        # Optionally compute distance matrix just like in training
        distance_matrix = compute_distance_matrix(
            batch["x_info"],
            normalize=True,
        )
        print(f"Distance matrix shape: {distance_matrix.shape}")

        # Plot this batch and print comparison
        plot_batch_with_raw(batch, raw_targets_batch, cfg, batch_idx, run_dir, scaler_dict)

if __name__ == "__main__":
    main()