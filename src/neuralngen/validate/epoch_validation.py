# src/neuralngen/validate/epoch_validation.py

import numpy as np
import torch
from collections import defaultdict
from neuralngen.dataset.hourlycamelsus import HourlyCamelsDataset
from neuralngen.utils.performance_measures import nse

def validate_epoch(model, cfg, device="cpu"):
    model.eval()

    dataset = HourlyCamelsDataset(cfg, is_train=False, period="validation")

    # Build index: basin -> list of dataset indices
    basin_to_indices = defaultdict(list)
    for i, (basin_id, start_idx) in enumerate(dataset.samples):
        basin_to_indices[basin_id].append(i)

    nse_per_basin = {}

    for basin in dataset.basins:
        indices = basin_to_indices[basin]
        if len(indices) == 0:
            continue

        x_d_list = []
        x_s_list = []
        y_list = []

        for idx in indices:
            data = dataset[idx]
            x_d_list.append(data["x_d"])
            x_s_list.append(data["x_s"])
            y_list.append(data["y"])

        y_true_all = []
        y_pred_all = []

        batch_size_val = 128

        for i in range(0, len(x_d_list), batch_size_val):
            x_d_chunk = torch.stack(x_d_list[i:i+batch_size_val]).to(device)
            x_s_chunk = torch.stack(x_s_list[i:i+batch_size_val]).to(device)
            y_chunk = torch.stack(y_list[i:i+batch_size_val]).to(device)

            with torch.no_grad():
                preds = model(x_d_chunk, x_s_chunk)
                y_hat_chunk = preds["y_hat"].cpu().numpy()

            y_true_chunk = y_chunk.cpu().numpy()

            y_true_all.append(y_true_chunk)
            y_pred_all.append(y_hat_chunk)

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)

        # flatten across time and batch
        y_true_flat = y_true_all.reshape(-1)
        y_pred_flat = y_pred_all.reshape(-1)

        nse_val = nse(y_true_flat, y_pred_flat)
        nse_per_basin[basin] = nse_val

        print(f"Basin {basin} NSE: {nse_val:.4f}")

    # Summary
    nse_values = np.array(list(nse_per_basin.values()))
    nse_values = nse_values[~np.isnan(nse_values)]

    if len(nse_values) > 0:
        print("\n======= Validation NSE Statistics =======")
        print(f"Mean NSE:   {np.mean(nse_values):.4f}")
        print(f"Median NSE: {np.median(nse_values):.4f}")
        print(f"Min NSE:    {np.min(nse_values):.4f}")
        print(f"Max NSE:    {np.max(nse_values):.4f}")
    else:
        print("No valid NSE values computed.")

    return nse_per_basin