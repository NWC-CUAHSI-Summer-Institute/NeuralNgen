# src/neuralngen/training/basetrainer.py

import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from neuralngen.validate.epoch_validation import validate_epoch
from neuralngen.utils.distance import compute_distance_matrix
from neuralngen.training.loss import ngenLoss
from neuralngen.dataset.collate import custom_collate
import math


# =====================================================================
# Basin Chronological Interleave Sampler (from Majid's approach)
# =====================================================================
class BasinChronoInterleaveSampler:
    """
    Yields batches that are chronological within each basin,
    but randomly interleaved across basins.

    Why this is necessary:
      Persistent LSTM carries (h, c) forward in time. If you shuffle all
      batch-units globally, you can process a later time-chunk before an
      earlier one for the same basin -> hidden-state time misalignment
      -> training collapses / bad NSE.

    How it works:
      - For each basin we have a sorted list of (basin, start_idx) pairs.
      - We keep a per-basin pointer indicating which batch comes next.
      - At each step we randomly choose from still-active basins and yield
        its NEXT chronological batch.
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

        # Precompute number of batches per basin
        self._basin_num_batches = {}
        for basin_id, pairs in self.basin_to_sorted_pairs.items():
            self._basin_num_batches[basin_id] = max(1, len(pairs) // self.batch_size)

    def set_epoch(self, epoch: int):
        """Call once per epoch so interleaving order changes."""
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        # Pointer to next chronological batch per basin
        ptr = {b: 0 for b in self._basin_num_batches}
        active = [b for b, nb in self._basin_num_batches.items() if nb > 0]

        while active:
            basin_id = rng.choice(active)
            k = ptr[basin_id]
            start = k * self.batch_size
            end = start + self.batch_size

            pairs = self.basin_to_sorted_pairs[basin_id]
            batch_pairs = pairs[start:end]

            if len(batch_pairs) < 2:
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
        """
        Parameters
        ----------
        cfg : Config object
            Configuration parameters.
        model : torch.nn.Module
            Your neural network model.
        dataset_class : Dataset class that takes (cfg, period) as arguments.
        """
        self.cfg = cfg
        self.model = model
        self.dataset_class = dataset_class

        self.device = torch.device(cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._set_seed()

        # Create run output directory
        self.run_dir = self._create_run_dir()

        # Load training dataset
        self.train_dataset = dataset_class(
            cfg,
            is_train=True,
            period="train",
            run_dir=self.run_dir,
            do_load_scalers=True
        )

        num_basins = len(getattr(self.train_dataset, "all_basins_with_samples", []))
        if num_basins == 0:
            raise RuntimeError("Training dataset contains no basins with time windows to sample.")

        total_windows = self.train_dataset.total_windows
        batch_size = self.cfg.batch_sites
        coverage_factor = self.cfg.epoch_coverage_factor

        self.num_train_batches_per_epoch = coverage_factor * math.ceil(total_windows / batch_size)

        print(f"Training will run for {self.num_train_batches_per_epoch} batches per epoch "
              f"({total_windows} total time windows across {num_basins} unique basins, "
              f"batch size {batch_size}, coverage factor {coverage_factor}).")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # Loss
        self.criterion = ngenLoss(self.cfg)

        # ----- Persistent training flag (Majid's approach) -----
        self.persistent_state = getattr(self.cfg, "persistent_state", False)

        if self.persistent_state:
            self._basin_sampler = self._build_basin_chrono_sampler()
            print(f"Persistent training enabled: chronological batching with "
                  f"hidden state carryover within each epoch.")
        else:
            self._basin_sampler = None

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

    # -----------------------------------------------------------------
    # Persistent training helpers (adapted from Majid's approach)
    # -----------------------------------------------------------------
    def _build_basin_chrono_sampler(self) -> BasinChronoInterleaveSampler:
        """
        Build mapping: basin_id -> list of (basin, start_idx) pairs,
        sorted chronologically by start_idx within each basin.

        This ensures within each basin, batches always go A1 -> A2 -> A3
        (chronological), even while basins are randomly interleaved.
        """
        basin_to_sorted_pairs: Dict[str, List[tuple]] = {}

        for basin, start_idx in self.train_dataset.samples:
            basin_to_sorted_pairs.setdefault(basin, []).append((basin, start_idx))

        # Sort by start_idx within each basin (chronological order)
        for basin_id in basin_to_sorted_pairs:
            basin_to_sorted_pairs[basin_id].sort(key=lambda x: x[1])

        return BasinChronoInterleaveSampler(
            basin_to_sorted_pairs=basin_to_sorted_pairs,
            batch_size=self.cfg.batch_sites,
            seed=self.cfg.seed if self.cfg.seed else 0,
        )

    @staticmethod
    def _flatten_time_batch(t: torch.Tensor) -> torch.Tensor:
        """
        Flatten [B, L, ...] -> [1, B*L, ...].

        The persistent LSTM expects a single continuous time axis for state
        carryover. By flattening across the batch dimension, we treat the
        batch as one long chronological segment.
        """
        if not torch.is_tensor(t) or t.dim() < 2:
            return t
        t = t.contiguous()
        b, l = t.shape[0], t.shape[1]
        return t.view(1, b * l, *t.shape[2:])

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    def train(self):
        """
        Train for a full pass over the dataset each epoch.
        When persistent_state=True, uses chronological basin batching
        with hidden state carryover (Majid's approach).
        """
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_losses = []

            if self.persistent_state:
                self._train_epoch_persistent(epoch, epoch_losses)
            else:
                self._train_epoch_standard(epoch, epoch_losses)

            # Epoch summary
            mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            print(f"Epoch {epoch} finished. Mean loss: {mean_loss:.4f}")

            # Save and validate
            self._save_model(epoch)
            validate_epoch(self.model, self.cfg, device=self.device, run_dir=self.run_dir)

    def _train_epoch_standard(self, epoch, epoch_losses):
        """Original training: random batches, no state carryover."""
        # 1) Shuffle once per epoch
        all_pairs = self.train_dataset.samples.copy()
        random.shuffle(all_pairs)

        # 2) Chunk into batches of size batch_sites
        batch_size = self.cfg.batch_sites
        batches = [all_pairs[i:i + batch_size]
                for i in range(0, len(all_pairs), batch_size)]

        # 3) Iterate over actual batches
        pbar = tqdm(batches, desc=f"Epoch {epoch}")
        for batch_pairs in pbar:
            # Build batch samples
            batch_samples = [
                self.train_dataset._load_window(basin, start_idx)
                for basin, start_idx in batch_pairs
            ]
            batch = custom_collate(batch_samples)

            # Move tensors to device
            x_d = batch["x_d"].to(self.device)
            x_s = batch["x_s"].to(self.device)
            y   = batch["y"].to(self.device)

            distance_matrix = compute_distance_matrix(batch["x_info"], normalize=True)

            # Forward (no hidden state)
            preds = self.model(x_d, x_s)
            seq_len = self.cfg.sequence_length
            y_hat = preds["y_hat"][..., seq_len:, :]
            y_true = y[..., seq_len:, :]

            loss, loss_components = self.criterion(
                prediction={"y_hat": y_hat},
                data={"y": y_true, "distance_matrix": distance_matrix}
            )

            self.optimizer.zero_grad()
            loss.backward()
            if getattr(self.cfg, "clip_gradient_norm", None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)
            self.optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

    def _train_epoch_persistent(self, epoch, epoch_losses):
        """
        Persistent training (Majid's approach):
        - Chronological batching per basin (BasinChronoInterleaveSampler)
        - Per-basin hidden state cache: {basin_id: (h, c)}
        - Hidden states carried across batches within this epoch
        - Cache is reset at epoch start (no cross-epoch persistence)
        - Time-batch flattened: [B, L, ...] -> [1, B*L, ...] for
          continuous time processing
        """
        # Reset hidden state cache at epoch start
        epoch_state_cache: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}

        # Update sampler interleaving for this epoch
        self._basin_sampler.set_epoch(epoch)

        pbar = tqdm(
            self._basin_sampler,
            desc=f"Epoch {epoch} (persistent)",
            total=len(self._basin_sampler),
        )

        for basin_id, batch_pairs in pbar:
            # Build batch samples (already chronological for this basin)
            batch_samples = [
                self.train_dataset._load_window(basin, start_idx)
                for basin, start_idx in batch_pairs
            ]
            batch = custom_collate(batch_samples)

            # Move to device
            x_d = batch["x_d"].to(self.device)
            x_s = batch["x_s"].to(self.device)
            y   = batch["y"].to(self.device)

            # Flatten time-batch: [B, L, ...] -> [1, B*L, ...]
            # This makes the model see one continuous chronological segment
            B, L = x_d.shape[0], x_d.shape[1]
            x_d_flat = self._flatten_time_batch(x_d)
            y_flat = self._flatten_time_batch(y)
            # Static features stay per-basin (take first sample)
            x_s_single = x_s[0:1, ...]

            # Load hidden state for this basin (None if first time this epoch)
            hidden = epoch_state_cache.get(basin_id, None)

            # Forward with carried hidden state
            preds = self.model(x_d_flat, x_s_single, hidden_state=hidden)

            # Update hidden state cache (detach to stop gradient flow across batches)
            new_hidden = preds.get("hidden_state", None)
            if new_hidden is not None:
                h, c = new_hidden
                epoch_state_cache[basin_id] = (h.detach(), c.detach())
            else:
                epoch_state_cache[basin_id] = None

            # Loss
            seg_len = B * L
            seq_len = self.cfg.sequence_length
            y_hat = preds["y_hat"][..., seq_len:, :]
            y_true = y_flat[..., seq_len:, :]

            # Build distance matrix from original (non-flattened) batch
            distance_matrix = compute_distance_matrix(batch["x_info"], normalize=True)

            loss, loss_components = self.criterion(
                prediction={"y_hat": y_hat},
                data={"y": y_true, "distance_matrix": distance_matrix}
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