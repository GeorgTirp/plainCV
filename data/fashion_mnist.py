# data/fashion_mnist.py
from typing import Iterator, Tuple

import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp


def _preprocess(example):
    image = example["image"]          # uint8 (28, 28, 1)
    label = example["label"]          # int64
    # We'll normalize in the model, but ensure dtype and shape here
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

        # Turn TF tensors into device arrays lazily
        for batch in tfds.as_numpy(ds):
            images = batch["image"]    # (B, 28,28,1), uint8
            labels = batch["label"]    # (B,)
            yield jnp.array(images), jnp.array(labels)

    return (
        make_split("train", shuffle=True),
        make_split("test", shuffle=False),
    )
