# src/neuralngen/training/persistent_trainer.py

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from neuralngen.dataset.collate import custom_collate
from neuralngen.training.basetrainer import BaseTrainer
from neuralngen.utils.distance import compute_distance_matrix


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


class PersistentTrainer(BaseTrainer):
    """
    Trainer with chronological batching and hidden-state carryover.

    Extends BaseTrainer by overriding the epoch training loop to:
    - Use BasinChronoInterleaveSampler for chronological, non-overlapping chunks
    - Cache and carry over LSTM hidden states across chunks within each epoch
    - Flatten multi-sample batches into a single continuous time sequence

    Reference: github.com/majid828/neuralhydrology-persistent
    """

    def __init__(self, cfg, model, dataset_class):
        super().__init__(cfg, model, dataset_class)

        # Build the chronological sampler and recalculate batches per epoch
        self._basin_sampler = self._build_basin_chrono_sampler()
        total_windows = self._persistent_total_windows()
        batch_size = 1  # persistent mode processes one basin-chunk at a time

        coverage_factor = self.cfg.epoch_coverage_factor
        self.num_train_batches_per_epoch = coverage_factor * math.ceil(total_windows / batch_size)

        num_basins = len(getattr(self.train_dataset, "all_basins_with_samples", []))
        print(
            f"[PersistentTrainer] Overriding to {self.num_train_batches_per_epoch} batches per epoch "
            f"({total_windows} total time windows across {num_basins} unique basins, "
            f"batch size {batch_size}, coverage factor {coverage_factor})."
        )
        print(
            "Persistent training enabled: chronological non-overlapping prediction chunks "
            "with hidden state carryover within each epoch."
        )

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
        """Reshape [B, L, ...] -> [1, B*L, ...] for continuous time processing."""
        if not torch.is_tensor(t) or t.dim() < 2:
            return t
        t = t.contiguous()
        b, l = t.shape[0], t.shape[1]
        return t.view(1, b * l, *t.shape[2:])

    def _train_epoch(self, epoch, epoch_losses):
        """Persistent training epoch: chronological chunks with hidden-state carryover."""
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
