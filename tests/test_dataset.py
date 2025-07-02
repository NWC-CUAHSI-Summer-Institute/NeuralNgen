# tests/test_dataset.py

from pathlib import Path
from neuralngen.dataset.hourlycamelsus import HourlyCamelsDataset
from neuralngen.utils.config import Config

cfg = Config(Path("configs/config.yml"))
dataset = HourlyCamelsDataset(cfg, is_train=True, period="train")
batch = dataset[0]
print(batch["x_d"].shape)
print(batch["y"].shape)