import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, cfg, model, dataset_class):
        """
        Parameters
        ----------
        cfg : Config object (or similar dict-like)
            Contains config parameters.
        model : torch.nn.Module
            Your LSTM model.
        dataset_class : Dataset class
            Dataset class that takes (cfg, period) as arguments.
        """
        self.cfg = cfg
        self.model = model
        self.dataset_class = dataset_class

        self.device = torch.device(cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Seed everything
        self._set_seed()

        # Prepare data
        self.train_dataset = dataset_class(cfg, period="train")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,  # each __getitem__ returns a full batch already
            num_workers=cfg.num_workers
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # Loss
        self.criterion = torch.nn.MSELoss()

        # Create output directory
        self.run_dir = self._create_run_dir()

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

            pbar = tqdm(self.train_loader, total=self.cfg.steps_per_epoch)
            for step, batch in enumerate(pbar, start=1):
                if step > self.cfg.steps_per_epoch:
                    break

                x_d = batch["x_d"].to(self.device)
                x_s = batch["x_s"].to(self.device)
                y = batch["y"].to(self.device)

                preds = self.model(x_d, x_s)
                loss = self.criterion(preds, y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)
                self.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": loss.item()})

            mean_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch} finished. Mean loss: {mean_epoch_loss:.4f}")

            self._save_model(epoch)

    def _save_model(self, epoch):
        path = self.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")