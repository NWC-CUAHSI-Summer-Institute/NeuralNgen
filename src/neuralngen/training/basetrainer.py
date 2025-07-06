# src/neuralngen/training/basetrainer.py

import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from neuralngen.validate.epoch_validation import validate_epoch
from neuralngen.utils.distance import compute_distance_matrix
from neuralngen.training.loss import ngenLoss

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
        self.train_dataset = dataset_class(cfg, is_train=True, period="train")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_sites,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # Loss
        self.criterion = ngenLoss(self.cfg)  # torch.nn.MSELoss()

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

            pbar = tqdm(self.train_loader)
            for step, batch in enumerate(pbar):

                x_d = batch["x_d"].to(self.device)
                x_s = batch["x_s"].to(self.device)
                y = batch["y"].to(self.device)

                distance_matrix = compute_distance_matrix(batch["x_info"], normalize=True)

                preds = self.model(x_d, x_s)

                # Warmup slicing
                sequence_length = self.cfg.sequence_length

                # keep only the portion after warm-up
                y_hat = preds["y_hat"][:, sequence_length:, :]
                y_true = y[:, sequence_length:, :]

                loss, loss_components = self.criterion(
                    prediction={"y_hat": y_hat},
                    data={"y": y_true, "distance_matrix": distance_matrix}
                )

                if step == 0:
                    print(f"\nLoss value BEFORE backward: {loss.item()}")

                if torch.isnan(loss):
                    print("!!! LOSS is NaN !!!")
                    # Optionally stop training to debug
                    raise RuntimeError("NaN loss detected")

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)
                self.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": loss.item()})

            mean_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch} finished. Mean loss: {mean_epoch_loss:.4f}")

            self._save_model(epoch)

            validate_epoch(self.model, self.cfg, device=self.device)

    def _save_model(self, epoch):
        path = self.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")