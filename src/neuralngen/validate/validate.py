# src/validate/validate.py

import argparse
import numpy as np
import torch
from pathlib import Path

from neuralngen.utils import Config
from neuralngen.models.ngenlstm import NgenLSTM
from neuralngen.dataset.hourlycamelsus import HourlyCamelsDataset
from neuralngen.utils.performance_measures import nse

def validate(cfg_path: Path, model_path: Path, device: str = "cpu"):
    # Load config
    cfg = Config(cfg_path)

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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Track NSE scores
    nse_per_basin = {}

    # Iterate through each basin in validation set
    basin_ids = dataset.basins
    for basin in basin_ids:
        # Collect all windows for this basin
        y_true_all = []
        y_pred_all = []

        # Gather all samples for this basin
        for idx, sample in enumerate(dataset.samples):
            sample_basin, start_idx = sample
            if sample_basin != basin:
                continue

            data = dataset[idx]
            x_d = data["x_d"].unsqueeze(0).to(device)      # [1, T, Din]
            x_s = data["x_s"].unsqueeze(0).to(device)      # [1, S]

            # Forward pass
            with torch.no_grad():
                preds = model(x_d, x_s)
                y_hat = preds["y_hat"].cpu().numpy().squeeze()  # [T]
            
            y_true = data["y"].cpu().numpy().squeeze()  # [T]
            
            y_true_all.append(y_true)
            y_pred_all.append(y_hat)

        if len(y_true_all) == 0:
            continue

        # Concatenate all windows
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)

        nse_val = nse(y_true_all, y_pred_all)
        nse_per_basin[basin] = nse_val
        print(f"Basin {basin} NSE: {nse_val:.4f}")

    # Report summary statistics
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
    args = parser.parse_args()

    validate(Path(args.config), Path(args.model), device=args.device)


if __name__ == "__main__":
    main()