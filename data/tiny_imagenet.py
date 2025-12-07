# data/tiny_imagenet.py
import os
from pathlib import Path
from typing import Iterator, Tuple, Optional

import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp

# Keep TF on CPU to avoid device conflicts with JAX.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


DATA_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
# By default, mirror the path TFDS was already using in the error message.
DEFAULT_DATA_ROOT = Path.home() / "tensorflow_datasets" / "tiny_imagenet"


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


def _find_dataset_dir(data_root: Path) -> Optional[Path]:
    """Try the common extraction locations for tiny-imagenet-200."""
    candidates = [
        data_root / "tiny-imagenet-200",
        data_root / "tiny_imagenet-200",
        data_root / "tiny-imagenet-200_extracted",
        data_root / "tiny-imagenet-200_extracted" / "tiny-imagenet-200",
    ]

    # Also search recursively for wnids.txt to be robust to user placements.
    candidates += [p.parent for p in data_root.rglob("wnids.txt")]

    for path in candidates:
        if not path.exists():
            continue
        if (path / "train").exists() and (path / "val").exists() and (path / "wnids.txt").exists():
            return path
    return None


def _ensure_dataset(data_root: Path) -> Path:
    """
    Make sure tiny-imagenet-200 is present on disk.

    We avoid relying on the (missing) TFDS builder and instead download the
    official archive if it's not already unpacked. If downloading is blocked,
    the raised error tells the user where to place the files manually.
    """
    existing = _find_dataset_dir(data_root)
    if existing:
        return existing

    data_root.mkdir(parents=True, exist_ok=True)
    try:
        tf.keras.utils.get_file(
            fname="tiny-imagenet-200.zip",
            origin=DATA_URL,
            cache_dir=str(data_root),
            cache_subdir="",
            extract=True,
            archive_format="zip",
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"tiny-imagenet-200 not found under {data_root}. "
            "Download the archive manually from "
            "http://cs231n.stanford.edu/tiny-imagenet-200.zip "
            f"and extract it into {data_root}."
        ) from exc

    existing = _find_dataset_dir(data_root)
    if not existing:
        raise FileNotFoundError(
            f"tiny-imagenet-200 failed to extract into a usable directory under {data_root}. "
            "Please download and unpack the archive manually."
        )
    return existing


def _load_metadata(dataset_dir: Path):
    wnids_path = dataset_dir / "wnids.txt"
    val_ann_path = dataset_dir / "val" / "val_annotations.txt"

    if not wnids_path.exists() or not val_ann_path.exists():
        raise FileNotFoundError(
            f"Missing wnids.txt or val_annotations.txt under {dataset_dir}. "
            "Make sure tiny-imagenet-200 is fully extracted."
        )

    wnids = [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

    val_labels = {}
    for line in val_ann_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            fname, wnid = parts[0], parts[1]
            val_labels[fname] = class_to_idx[wnid]

    return class_to_idx, val_labels


def _make_lookup_table(mapping: dict) -> tf.lookup.StaticHashTable:
    keys = tf.constant(list(mapping.keys()), dtype=tf.string)
    values = tf.constant(list(mapping.values()), dtype=tf.int64)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=-1,
    )


def _make_dataset(
    file_pattern: str,
    label_fn,
    batch_size: int,
    shuffle: bool,
    seed: int,
    image_size: int,
):
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle, seed=seed)

    def _load_and_preprocess(path: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        label = label_fn(path)
        example = _preprocess({"image": image, "label": label}, image_size=image_size)
        return example["image"], example["label"]

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(50_000, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    return ds


def get_datasets(
    batch_size: int,
    seed: int = 0,
    image_size: Optional[int] = 64,
    data_root: Optional[Path | str] = None,
) -> Tuple[
    Iterator[Tuple[jnp.ndarray, jnp.ndarray]],
    Iterator[Tuple[jnp.ndarray, jnp.ndarray]],
]:
    """
    Return train & val iterators over TinyImageNet (default 64x64, 3 channels, 200 classes).

    We implement a lightweight loader that works from the official archive
    instead of relying on the unavailable TFDS builder.
    """
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    dataset_dir = _ensure_dataset(root)
    class_to_idx, val_labels = _load_metadata(dataset_dir)

    class_table = _make_lookup_table(class_to_idx)
    val_table = _make_lookup_table(val_labels)

    train_pattern = str(dataset_dir / "train" / "*" / "images" / "*.JPEG")
    val_pattern = str(dataset_dir / "val" / "images" / "*.JPEG")

    def train_label_from_path(path: tf.Tensor):
        parts = tf.strings.split(path, os.sep)
        wnid = parts[-3]  # .../train/<wnid>/images/<file>.JPEG
        return class_table.lookup(wnid)

    def val_label_from_path(path: tf.Tensor):
        fname = tf.strings.split(path, os.sep)[-1]
        return val_table.lookup(fname)

    def make_split(split: str, shuffle: bool):
        pattern = train_pattern if split == "train" else val_pattern
        label_fn = train_label_from_path if split == "train" else val_label_from_path
        ds = _make_dataset(
            file_pattern=pattern,
            label_fn=label_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            image_size=image_size,
        )
        for images, labels in tfds.as_numpy(ds):
            yield jnp.array(images), jnp.array(labels)

    return (
        make_split("train", shuffle=True),
        make_split("validation", shuffle=False),
    )
