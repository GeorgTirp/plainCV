from .fashion_mnist import get_datasets as _get_fashion_mnist
from .tiny_imagenet import get_datasets as _get_tiny_imagenet
from . import datasets as datasets


def get_datasets(dataset: str, batch_size: int, seed: int = 0, image_size: int | None = None):
    """
    Dispatch to the requested dataset.

    Args:
        dataset: one of {"fashion_mnist", "tiny_imagenet"} (case-insensitive).
        batch_size: batch size for both train/test iterators.
        seed: RNG seed for shuffling.
        image_size: optional resize; TinyImageNet defaults to 64 if None.
    """
    name = dataset.lower()
    if name == "fashion_mnist":
        return _get_fashion_mnist(batch_size=batch_size, seed=seed)
    elif name in {"tiny_imagenet", "tiny-imagenet", "tinyimagenet"}:
        return _get_tiny_imagenet(batch_size=batch_size, seed=seed, image_size=image_size)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: fashion_mnist, tiny_imagenet.")


__all__ = ["get_datasets", "datasets"]
