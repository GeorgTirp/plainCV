# data/tiny_imagenet.py
from typing import Iterator, Tuple, Optional

import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp

# Keep TF on CPU to avoid device conflicts with JAX.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def _preprocess(example, image_size: int):
    image = example["image"]  # uint8 (64, 64, 3)
    label = example["label"]  # int64

    if image_size and image_size != 64:
        image = tf.image.resize(image, (image_size, image_size), method="bilinear")
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)

    return {
        "image": image,
        "label": label,
    }


def get_datasets(
    batch_size: int,
    seed: int = 0,
    image_size: Optional[int] = 64,
) -> Tuple[
    Iterator[Tuple[jnp.ndarray, jnp.ndarray]],
    Iterator[Tuple[jnp.ndarray, jnp.ndarray]],
]:
    """
    Return train & val iterators over TinyImageNet (default 64x64, 3 channels, 200 classes).
    """
    ds_builder = tfds.builder("tiny_imagenet")
    ds_builder.download_and_prepare()

    def make_split(split: str, shuffle: bool) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        ds = ds_builder.as_dataset(split=split)
        ds = ds.map(lambda ex: _preprocess(ex, image_size=image_size))
        if shuffle:
            ds = ds.shuffle(50_000, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(1)

        for batch in tfds.as_numpy(ds):
            images = batch["image"]
            labels = batch["label"]
            yield jnp.array(images), jnp.array(labels)

    return (
        make_split("train", shuffle=True),
        make_split("validation", shuffle=False),
    )
