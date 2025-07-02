import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray
import traceback

from NeuralNGEN.src.utils.config import Config
from NeuralNGEN.src.dataset import HourlyCamelsUS
from NeuralNGEN.src.dataset.batch import SpatialTemporalBatcher

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # ✅ Set your paths
        data_dir = Path("/Users/mohammad/Desktop/PHD/PHD/CIROH_SUMMER_SCHOOL/CAMELS_data_sample")
        basin_id = None  # or a list like ["12010000"]
        hourly_forcing_name = "aorc_hourly"

        # ✅ Define Config without spatial/temporal_length
        cfg = Config({
            'data_dir': data_dir,
            'forcings': [hourly_forcing_name],
            'target_variables': ['QObs(mm/h)'],
            'dynamic_inputs': ['APCP_surface', 'DLWRF_surface'],
            'static_attributes': ['elev_mean', 'area_gages2'],
            'train_basin_file': str('/Users/mohammad/Desktop/PHD/PHD/CIROH_SUMMER_SCHOOL/train_basins.txt'),
            'train_start_date': '29/09/1993',
            'train_end_date': '03/10/1995',
            'seq_length': 24,
            'predict_last_n': 1,
            'train_dir': Path('./runs/train'),
            'loss': 'mse',
            'model': 'lstm'
        })

        print(f"\n⏳ Loading data for basin: {basin_id}")
        dataset = HourlyCamelsUS(cfg, is_train=True, period='train', basin=basin_id)
        print(f"✅ Dataset loaded with {len(dataset)} samples.")

        # Optional: View a sample
        sample = dataset[0]
        print("\n🔹 Sample from first basin:")
        print(sample)

        # -------------------------------
        # ✅ Spatial-Temporal Batching
        # -------------------------------
        print("\n⏳ Running SpatialTemporalBatcher...")

        raw_samples = [dataset[i] for i in range(len(dataset))]
        spatial_length = 7
        temporal_length = 200  # One week hourly

        batcher = SpatialTemporalBatcher(
            all_pre_sliced_samples=raw_samples,
            spatial_length=spatial_length,
            temporal_length_batcher=temporal_length,
            seed=42
        )
        all_batches = batcher.get_all_batches()
        print(f"✅ Generated {len(all_batches)} spatial-temporal batches.")

        # Show shape from first 2 batches
        for i, data_dict in enumerate(all_batches[:2]):
            print(f"\n🔹 Batch {i + 1}")
            print("x_d:", data_dict['x_d'].shape)  # [B, T, D]
            print("y:", data_dict['y'].shape)      # [B, T, 1]
            print("date:", data_dict['date'].shape)
            print("Clusters:", data_dict['Clusters'])

    except IndexError:
        print("\n❌ No valid sequences were found.")
    except Exception as e:
        print("\n❌ An error occurred:")
        traceback.print_exc()