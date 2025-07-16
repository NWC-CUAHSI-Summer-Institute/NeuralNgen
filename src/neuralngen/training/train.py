# src/neuralngen/training/train.py

import argparse
from pathlib import Path

from neuralngen.models.ngenlstm import NgenLSTM
from neuralngen.dataset.hourlycamelsus import HourlyCamelsDataset
from neuralngen.dataset.dailycamelsus import DailyCamelsDataset
from neuralngen.training.basetrainer import BaseTrainer
from neuralngen.utils import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Load config
    cfg = Config(Path(args.config))

    # Initialize dataset just once to get data shapes
    if cfg.dataset == "hourly_camels_us":
        dataset = HourlyCamelsDataset(cfg, is_train=True, period="train", do_load_scalers=False)
    if cfg.dataset == "daily_camels_us":
        dataset = DailyCamelsDataset(cfg, is_train=True, period="train", do_load_scalers=False)

    # Sample batch
    batch = dataset[0]
    x_d = batch["x_d"]           # [B, T, Din]
    x_s = batch["x_s"]           # [B, S]
    x_info = batch["x_info"]
    y = batch["y"]               # [B, T, Dout]

    model = NgenLSTM(
        dynamic_input_size=x_d.shape[-1],
        static_input_size=x_s.shape[-1],
        hidden_size=cfg.hidden_size,
        output_size=y.shape[-1],
    )

    # Plug into the trainer
    if cfg.dataset == "hourly_camels_us":
        trainer = BaseTrainer(cfg, model, HourlyCamelsDataset)
    if cfg.dataset == "daily_camels_us":
        trainer = BaseTrainer(cfg, model, DailyCamelsDataset)
    trainer.train()


if __name__ == "__main__":
    main()