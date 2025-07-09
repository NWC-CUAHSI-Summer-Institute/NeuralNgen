# src/validate/validate.py

import argparse
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from neuralngen.utils import Config
from neuralngen.models.ngenlstm import NgenLSTM
from neuralngen.dataset.hourlycamelsus import HourlyCamelsDataset
from neuralngen.utils.performance_measures import nse


def validate_single_basin(basin, cfg_path, model_path, device_str):
    """
    Run validation on a single basin. Separate process.

    Returns
    -------
    tuple (basin_id, nse_value)
    """

    # Load config
    cfg = Config(Path(cfg_path))

    # Prepare dataset
    dataset = HourlyCamelsDataset(cfg, is_train=False, period="validation")

    # Build model
    batch = dataset[0]
    model = NgenLSTM(
        dynamic_input_size=batch["x_d"].shape[-1],
        static_input_size=batch["x_s"].shape[-1],
        hidden_size=cfg.hidden_size,
        output_size=batch["y"].shape[-1],
    )
    model.load_state_dict(torch.load(model_path, map_location=device_str))
    model.to(device_str)
    model.eval()

    y_true_all = []
    y_pred_all = []

    for idx, sample in enumerate(dataset.samples):
        sample_basin, start_idx = sample
        if sample_basin != basin:
            continue

        data = dataset[idx]
        x_d = data["x_d"].unsqueeze(0).to(device_str)
        x_s = data["x_s"].unsqueeze(0).to(device_str)

        with torch.no_grad():
            preds = model(x_d, x_s)
            y_hat = preds["y_hat"].cpu().numpy().squeeze()

        y_true = data["y"].cpu().numpy().squeeze()

        y_true_all.append(y_true)
        y_pred_all.append(y_hat)

    if len(y_true_all) == 0:
        return basin, np.nan

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    nse_val = nse(y_true_all, y_pred_all)
    return basin, nse_val


def validate(cfg_path: Path, model_path: Path, device: str = "cpu", max_workers: int = 4):
    cfg = Config(cfg_path)

    # Load once to get basin list
    dataset = HourlyCamelsDataset(cfg, is_train=False, period="validation")
    basin_ids = dataset.basins

    nse_per_basin = {}

    # Parallel execution
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for basin in basin_ids:
            f = executor.submit(
                validate_single_basin,
                basin,
                str(cfg_path),
                str(model_path),
                device
            )
            futures.append(f)

        for f in as_completed(futures):
            basin, nse_val = f.result()
            nse_per_basin[basin] = nse_val
            if not np.isnan(nse_val):
                print(f"Basin {basin} NSE: {nse_val:.4f}")

    # Summary
    nse_values = np.array(list(nse_per_basin.values()))
    nse_values = nse_values[~np.isnan(nse_values)]

    if len(nse_values) > 0:
        print("\n======= NSE Statistics =======")
        print(f"Mean NSE: {np.mean(nse_values):.4f}")
        print(f"Median NSE: {np.median(nse_values):.4f}")
        print(f"Min NSE: {np.min(nse_values):.4f}")
        print(f"Max NSE: {np.max(nse_values):.4f}")
    else:
        print("No valid NSE values computed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model .pt file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of processes for parallel validation."
    )
    args = parser.parse_args()

    validate(Path(args.config), Path(args.model), device=args.device, max_workers=args.max_workers)


if __name__ == "__main__":
    main()