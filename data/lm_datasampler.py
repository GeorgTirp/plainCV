"""
JAX-friendly samplers for use with torch.utils.data.DataLoader.

- No torch.distributed dependency.
- Multi-process sharding uses jax.process_index() / jax.process_count().
- Supports resuming via start_idx/start_iter like your original samplers.
"""

from __future__ import annotations

from typing import Sized, Optional, Iterator, List
import math
import numpy as np
import jax

from torch.utils.data import Sampler


class StatefulSequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order, with optional start offset."""

    def __init__(self, data_source: Sized, batch_size: int, start_idx: int = 0):
        self.data_source = data_source
        self.start = int(start_idx) * int(batch_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, len(self.data_source)))

    def __len__(self) -> int:
        return max(0, len(self.data_source) - self.start)


class StatefulRandomSampler(Sampler[int]):
    """
    Samples elements either sequentially or shuffled, with optional start offset.
    If shuffle=True, sampling order is deterministic given seed and advances across epochs
    (each __iter__ call consumes RNG state).
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        start_idx: int = 0,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.start = int(start_idx) * int(batch_size)
        self.shuffle = bool(shuffle)

        if self.shuffle:
            if seed is None:
                raise ValueError("Seed must be set if shuffle=True in a stateful sampler.")
            # persistent RNG: each __iter__ yields a new permutation deterministically
            self.rng = np.random.default_rng(int(seed))
        else:
            self.rng = None

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        indices = np.arange(n, dtype=np.int64)
        if self.shuffle:
            indices = self.rng.permutation(indices)
        return iter(indices[self.start :].tolist())

    def __len__(self) -> int:
        return max(0, len(self.data_source) - self.start)


class StatefulJaxDistributedSampler(Sampler[int]):
    """
    JAX replacement for StatefulDistributedSampler.

    Each JAX process gets a distinct contiguous block of indices:
      [rank*num_samples, (rank+1)*num_samples)

    Then (optionally) shuffles *within the rank* using (epoch + seed),
    and supports resuming from start_iter * batch_size.
    """

    def __init__(
        self,
        dataset: Sized,
        batch_size: int,
        seed: int = 0,
        start_iter: int = 0,
        shuffle_within_rank: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.seed = int(seed)
        self.start_iter = int(start_iter)
        self.shuffle_within_rank = bool(shuffle_within_rank)
        self.drop_last = bool(drop_last)

        self.rank = jax.process_index()
        self.num_replicas = jax.process_count()
        self.epoch = 0

        n = len(dataset)

        if self.drop_last:
            self.total_size = n - (n % self.num_replicas)
        else:
            # pad up to be divisible
            self.total_size = int(math.ceil(n / self.num_replicas) * self.num_replicas)

        self.num_samples = self.total_size // self.num_replicas

        # contiguous assignment per process (matches your original logic)
        self.rank_start = self.rank * self.num_samples
        self.rank_end = self.rank_start + self.num_samples

        print(f"rank {self.rank}: sampler created, start_iter={self.start_iter}, num_samples={self.num_samples}")

    def set_epoch(self, epoch: int) -> None:
        """Call this at the start of each epoch if you want epoch-dependent shuffles."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        # base contiguous block for this rank
        indices = np.arange(self.rank_start, self.rank_end, dtype=np.int64)

        # If not drop_last, we may have indices beyond len(dataset). Wrap/pad.
        n = len(self.dataset)
        if not self.drop_last:
            indices = indices % n

        if self.shuffle_within_rank:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm = rng.permutation(self.num_samples)
            indices = indices[perm]

        # resume offset in units of batches
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]

        return iter(indices.tolist())

    def __len__(self) -> int:
        start_index = self.start_iter * self.batch_size
        return max(0, self.num_samples - start_index)


class DeterministicJaxDistributedSampler(StatefulJaxDistributedSampler):
    """Same as above but with no shuffle (fixed assignment per rank)."""

    def __init__(self, dataset: Sized, batch_size: int, start_iter: int = 0, drop_last: bool = True):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            seed=0,
            start_iter=start_iter,
            shuffle_within_rank=False,
            drop_last=drop_last,
        )
        print(f"rank {self.rank}: DEBUGGING sampler created...")
