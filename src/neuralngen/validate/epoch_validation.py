# src/validate/epoch_validation.py

import numpy as np
import torch
from neuralngen.dataset.hourlycamelsus import HourlyCamelsDataset
from neuralngen.utils.performance_measures import nse

def validate_epoch(model, cfg, device="cpu"):
    """
    Validate model on the validation basins and print NSE statistics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    cfg : Config
        Configuration object.
    device : str
        Device for inference.
    """
    model.eval()

    # Load validation dataset
    dataset = HourlyCamelsDataset(cfg, is_train=False, period="validation")

    basin_ids = dataset.basins
    nse_per_basin = {}

    for basin in basin_ids:
        y_true_all = []
        y_pred_all = []

        for idx, sample in enumerate(dataset.samples):
            sample_basin, start_idx = sample
            if sample_basin != basin:
                continue

            data = dataset[idx]
            x_d = data["x_d"].unsqueeze(0).to(device)    # [1, T, Din]
            x_s = data["x_s"].unsqueeze(0).to(device)    # [1, S]

            with torch.no_grad():
                preds = model(x_d, x_s)
                y_hat = preds["y_hat"].cpu().numpy().squeeze()

            y_true = data["y"].cpu().numpy().squeeze()

            y_true_all.append(y_true)
            y_pred_all.append(y_hat)

        if len(y_true_all) == 0:
            continue

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)

        nse_val = nse(y_true_all, y_pred_all)
        nse_per_basin[basin] = nse_val

    # Summary statistics
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