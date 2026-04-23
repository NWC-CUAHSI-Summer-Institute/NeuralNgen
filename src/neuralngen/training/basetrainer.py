# src/neuralngen/training/basetrainer.py

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from neuralngen.dataset.collate import custom_collate
from neuralngen.training.loss import ngenLoss
from neuralngen.utils.distance import compute_distance_matrix
from neuralngen.validate.epoch_validation import validate_epoch


class BaseTrainer:
    """Standard trainer with random-shuffle batching (no hidden-state carryover)."""

    def __init__(self, cfg, model, dataset_class):
        self.cfg = cfg
        self.model = model
        self.dataset_class = dataset_class

        self.device = torch.device(
            cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self._set_seed()

        self.run_dir = self._create_run_dir()

        self.train_dataset = dataset_class(
            cfg,
            is_train=True,
            period="train",
            run_dir=self.run_dir,
            do_load_scalers=True,
        )

        num_basins = len(getattr(self.train_dataset, "all_basins_with_samples", []))
        if num_basins == 0:
            raise RuntimeError("Training dataset contains no basins with time windows to sample.")

        total_windows = self.train_dataset.total_windows
        batch_size = self.cfg.batch_sites

        coverage_factor = self.cfg.epoch_coverage_factor
        self.num_train_batches_per_epoch = coverage_factor * math.ceil(total_windows / batch_size)

        print(
            f"Training will run for {self.num_train_batches_per_epoch} batches per epoch "
            f"({total_windows} total time windows across {num_basins} unique basins, "
            f"batch size {batch_size}, coverage factor {coverage_factor})."
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.criterion = ngenLoss(self.cfg)

    def _set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def _create_run_dir(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(self.cfg.output_dir) / f"{self.cfg.experiment_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def train(self):
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_losses = []

            self._train_epoch(epoch, epoch_losses)

            mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            print(f"Epoch {epoch} finished. Mean loss: {mean_loss:.4f}")

            self._save_model(epoch)
            validate_epoch(self.model, self.cfg, device=self.device, run_dir=self.run_dir)

    def _train_epoch(self, epoch, epoch_losses):
        """Run one training epoch. Override in subclasses for different strategies."""
        all_pairs = self.train_dataset.samples.copy()
        random.shuffle(all_pairs)

        batch_size = self.cfg.batch_sites
        batches = [all_pairs[i:i + batch_size] for i in range(0, len(all_pairs), batch_size)]

        pbar = tqdm(batches, desc=f"Epoch {epoch}")
        for batch_pairs in pbar:
            batch_samples = [
                self.train_dataset._load_window(basin, start_idx)
                for basin, start_idx in batch_pairs
            ]
            batch = custom_collate(batch_samples)

            x_d = batch["x_d"].to(self.device)
            x_s = batch["x_s"].to(self.device)
            y = batch["y"].to(self.device)

            distance_matrix = compute_distance_matrix(batch["x_info"], normalize=True)

            preds = self.model(x_d, x_s)
            seq_len = self.cfg.sequence_length
            y_hat = preds["y_hat"][..., seq_len:, :]
            y_true = y[..., seq_len:, :]

            loss, loss_components = self.criterion(
                prediction={"y_hat": y_hat},
                data={"y": y_true, "distance_matrix": distance_matrix},
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

    def _save_model(self, epoch):
        path = self.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")
