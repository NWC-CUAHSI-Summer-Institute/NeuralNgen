# tests/test_dataset.py

from pathlib import Path
from neuralngen.dataset.hourlycamelsus import HourlyCamelsDataset
from neuralngen.utils.config import Config

cfg = Config(Path("configs/config.yml"))
dataset = HourlyCamelsDataset(cfg, is_train=True, period="train")
batch = dataset[0]

print("x_d")
print(batch["x_d"].shape)
print(batch["x_d"])

print("x_info")
print(len(batch["x_info"]))
print(batch["x_info"])

print("x_s")
print(batch["x_s"].shape)
print(batch["x_s"])

print("y")
print(batch["y"].shape)
print(batch["y"])