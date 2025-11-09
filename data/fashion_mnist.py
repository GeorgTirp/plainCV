# data/fashion_mnist.py
from typing import Iterator, Tuple

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp

# --- NEW: disable TF using the GPU, it should stay CPU-only ---
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    # If TF is older / API mismatch, just ignore.
    pass
# ---------------------------------------------------------------


def _preprocess(example):
    image = example["image"]          # uint8 (28, 28, 1)
    label = example["label"]          # int64
    return {
        "image": image,
        "label": label,
    }


def get_datasets(batch_size: int, seed: int = 0):
    """Return train & test iterators over Fashion-MNIST."""
    ds_builder = tfds.builder("fashion_mnist")
    ds_builder.download_and_prepare()

    def make_split(split: str, shuffle: bool) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        ds = ds_builder.as_dataset(split=split)
        ds = ds.map(_preprocess)
        if shuffle:
            ds = ds.shuffle(10_000, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(1)

        for batch in tfds.as_numpy(ds):
            images = batch["image"]
            labels = batch["label"]
            yield jnp.array(images), jnp.array(labels)

    return (
        make_split("train", shuffle=True),
        make_split("test", shuffle=False),
    )
