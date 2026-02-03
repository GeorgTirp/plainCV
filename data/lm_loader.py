import jax
import jax.numpy as jnp
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from data.lm_datasampler import (
    StatefulSequentialSampler,
    StatefulRandomSampler,
    StatefulJaxDistributedSampler,
    DeterministicJaxDistributedSampler,
)



def get_dataloaders(cfg):
    train_set = load_from_disk(cfg.trainset_path)
    if not isinstance(train_set, Dataset):
        raise ValueError("dataset should be a datasets.Dataset")

    # ---- JAX multi-process sharding (important) ----
    if jax.process_count() > 1:
        train_set = train_set.shard(
            num_shards=jax.process_count(),
            index=jax.process_index(),
            contiguous=True,
        )

    train_sampler = _get_sampler_jax(train_set, cfg)

    def collate_fn(batch):
        # Return numpy (or torch). JAX is happy with numpy.
        input_ids = np.stack([np.asarray(x["input_ids"]) for x in batch], axis=0)
        out = {"input_ids": input_ids}
        if "docs_lengths" in batch[0]:
            out["docs_lengths"] = [np.asarray(x["docs_lengths"]).tolist() for x in batch]
        return out

    trainloader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=cfg.micro_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,  # strongly recommended for JAX pmaps
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=True if cfg.num_workers > 0 else False,
        collate_fn=collate_fn if "docs_lengths" in train_set.column_names else None,
    )

    validloader = None
    if cfg.validset_path:
        valid_set = load_from_disk(cfg.validset_path)
        if not isinstance(valid_set, Dataset):
            raise ValueError("dataset should be a datasets.Dataset")

        if getattr(cfg, "valid_tokens", False):
            valid_rows = cfg.valid_tokens // (cfg.seq_len + 1)
            if valid_rows > 0:
                valid_rows = min(valid_rows, len(valid_set))
                valid_set = valid_set.take(valid_rows)

        if jax.process_count() > 1:
            valid_set = valid_set.shard(
                num_shards=jax.process_count(),
                index=jax.process_index(),
                contiguous=True,
            )

        valid_sampler = SequentialSampler(valid_set)
        validloader = DataLoader(
            valid_set,
            batch_size=cfg.micro_batch_size,
            drop_last=True,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            persistent_workers=False,
            collate_fn=collate_fn if "docs_lengths" in valid_set.column_names else None,
        )

    return trainloader, validloader


def _get_sampler_jax(train_set, cfg):
    # JAX sharding already handled above, so no DDP sampler needed.
    if cfg.sampler == "random":
        return RandomSampler(
            train_set,
            generator=torch.Generator().manual_seed(cfg.sampler_seed) if cfg.sampler_seed else None,
        )

    elif cfg.sampler == "sequential":
        return SequentialSampler(train_set)

    elif cfg.sampler == "stateful_random":
        micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
        return StatefulRandomSampler(
            train_set,
            batch_size=cfg.micro_batch_size,
            shuffle=True,
            seed=cfg.sampler_seed,
            start_idx=micro_step_start,
        )

    elif cfg.sampler == "stateful_sequential":
        micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
        return StatefulSequentialSampler(
            train_set,
            batch_size=cfg.micro_batch_size,
            start_idx=micro_step_start,
        )

    else:
        raise NotImplementedError(f"Sampler {cfg.sampler} is not implemented.")
