# src/neuralngen/training/basetrainer.py

import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from neuralngen.dataset.collate import custom_collate
from neuralngen.training.loss import ngenLoss
from neuralngen.utils.distance import compute_distance_matrix
from neuralngen.validate.epoch_validation import validate_epoch


class BasinChronoInterleaveSampler:
    """
    Yields batches that are chronological within each basin,
    but randomly interleaved across basins.
    """

    def __init__(
        self,
        basin_to_sorted_pairs: Dict[str, List[tuple]],
        batch_size: int,
        seed: int = 0,
    ):
        self.basin_to_sorted_pairs = basin_to_sorted_pairs
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0

        self._basin_num_batches = {}
        for basin_id, pairs in self.basin_to_sorted_pairs.items():
            self._basin_num_batches[basin_id] = max(1, math.ceil(len(pairs) / self.batch_size))

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        ptr = {b: 0 for b in self._basin_num_batches}
        active = [b for b, nb in self._basin_num_batches.items() if nb > 0]

        while active:
            basin_id = rng.choice(active)
            k = ptr[basin_id]
            start = k * self.batch_size
            end = start + self.batch_size

            pairs = self.basin_to_sorted_pairs[basin_id]
            batch_pairs = pairs[start:end]

            if not batch_pairs:
                ptr[basin_id] += 1
                if ptr[basin_id] >= self._basin_num_batches[basin_id]:
                    active.remove(basin_id)
                continue

            yield basin_id, batch_pairs

            ptr[basin_id] += 1
            if ptr[basin_id] >= self._basin_num_batches[basin_id]:
                active.remove(basin_id)

    def __len__(self):
        return sum(self._basin_num_batches.values())


class BaseTrainer:
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

        self.persistent_state = getattr(self.cfg, "persistent_state", False)

        if self.persistent_state:
            self._basin_sampler = self._build_basin_chrono_sampler()
            total_windows = self._persistent_total_windows()
            batch_size = 1
        else:
            self._basin_sampler = None
            total_windows = self.train_dataset.total_windows
            batch_size = self.cfg.batch_sites

        coverage_factor = self.cfg.epoch_coverage_factor
        self.num_train_batches_per_epoch = coverage_factor * math.ceil(total_windows / batch_size)

        print(
            f"Training will run for {self.num_train_batches_per_epoch} batches per epoch "
            f"({total_windows} total time windows across {num_basins} unique basins, "
            f"batch size {batch_size}, coverage factor {coverage_factor})."
        )

        if self.persistent_state:
            print(
                "Persistent training enabled: chronological non-overlapping prediction chunks "
                "with hidden state carryover within each epoch."
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

    def _build_basin_chrono_sampler(self) -> BasinChronoInterleaveSampler:
        """
        Build mapping using non-overlapping prediction chunks.
        Chunk length is (sequence_length + batch_timesteps), and each new chunk
        starts every batch_timesteps so predicted segments do not overlap.
        """
        basin_to_sorted_pairs: Dict[str, List[tuple]] = {}
        step = self.cfg.batch_timesteps

        for basin in self.train_dataset.all_basins_with_samples:
            n_steps = self.train_dataset.dynamic_inputs[basin].shape[0]
            max_start = n_steps - self.train_dataset.window_size
            starts = range(0, max_start + 1, step)
            basin_to_sorted_pairs[basin] = [(basin, start_idx) for start_idx in starts]

        for basin_id in basin_to_sorted_pairs:
            basin_to_sorted_pairs[basin_id].sort(key=lambda x: x[1])

        return BasinChronoInterleaveSampler(
            basin_to_sorted_pairs=basin_to_sorted_pairs,
            batch_size=self.cfg.batch_sites,
            seed=self.cfg.seed if self.cfg.seed else 0,
        )

    def _persistent_total_windows(self) -> int:
        return sum(len(pairs) for pairs in self._basin_sampler.basin_to_sorted_pairs.values())

    @staticmethod
    def _flatten_time_batch(t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t) or t.dim() < 2:
            return t
        t = t.contiguous()
        b, l = t.shape[0], t.shape[1]
        return t.view(1, b * l, *t.shape[2:])

    def train(self):
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_losses = []

            if self.persistent_state:
                self._train_epoch_persistent(epoch, epoch_losses)
            else:
                self._train_epoch_standard(epoch, epoch_losses)

            mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            print(f"Epoch {epoch} finished. Mean loss: {mean_loss:.4f}")

            self._save_model(epoch)
            validate_epoch(self.model, self.cfg, device=self.device, run_dir=self.run_dir)

    def _train_epoch_standard(self, epoch, epoch_losses):
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

    def _train_epoch_persistent(self, epoch, epoch_losses):
        epoch_state_cache: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}

        self._basin_sampler.set_epoch(epoch)

        pbar = tqdm(
            self._basin_sampler,
            desc=f"Epoch {epoch} (persistent)",
            total=len(self._basin_sampler),
        )

        for basin_id, batch_pairs in pbar:
            batch_samples = [
                self.train_dataset._load_window(basin, start_idx)
                for basin, start_idx in batch_pairs
            ]
            batch = custom_collate(batch_samples)

            x_d = batch["x_d"].to(self.device)
            x_s = batch["x_s"].to(self.device)
            y = batch["y"].to(self.device)

            # Chunks are chronological and their predicted portions do not overlap.
            B, L = x_d.shape[0], x_d.shape[1]
            x_d_flat = self._flatten_time_batch(x_d)
            y_flat = self._flatten_time_batch(y)
            x_s_single = x_s[0:1, ...]

            hidden = epoch_state_cache.get(basin_id, None)

            preds = self.model(x_d_flat, x_s_single, hidden_state=hidden)

            new_hidden = preds.get("hidden_state", None)
            if new_hidden is not None:
                h, c = new_hidden
                epoch_state_cache[basin_id] = (h.detach(), c.detach())
            else:
                epoch_state_cache[basin_id] = None

            seq_len = self.cfg.sequence_length
            y_hat = preds["y_hat"][..., seq_len:, :]
            y_true = y_flat[..., seq_len:, :]

            distance_matrix = compute_distance_matrix(batch["x_info"], normalize=True)

            loss, loss_components = self.criterion(
                prediction={"y_hat": y_hat},
                data={"y": y_true, "distance_matrix": distance_matrix},
            )

            if torch.isnan(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            if getattr(self.cfg, "clip_gradient_norm", None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)
            self.optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item(), "basin": basin_id})

    def _save_model(self, epoch):
        path = self.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")
