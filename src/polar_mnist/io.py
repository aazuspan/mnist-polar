from __future__ import annotations

from importlib import resources
import struct

from numpy.typing import NDArray
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

from polar_mnist.transforms import Normalize1D, ToTensor1D, PolarAggregation


def _read_encoded_images(module: str, file_name: str) -> NDArray:
    """
    Read the encoded images from the given file path.

    https://stackoverflow.com/a/53181925
    """
    with resources.files(module).joinpath(file_name).open("rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))
        return data.reshape((size, nrows, ncols))


def _read_encoded_labels(module: str, file_name: str) -> NDArray:
    """
    Read the encoded images from the given file path.

    https://stackoverflow.com/a/53181925
    """
    with resources.files(module).joinpath(file_name).open("rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))
        return data.reshape((size,))


def load_mnist() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Load the MNIST images and labels from the resources into Numpy arrays.

    Returns a tuple of four numpy arrays:
    - train_images: The training images.
    - train_labels: The training labels.
    - test_images: The test images.
    - test_labels: The test labels.
    """
    train_images = _read_encoded_images("polar_mnist.data", "train-images-idx3-ubyte")
    train_labels = _read_encoded_labels("polar_mnist.data", "train-labels-idx1-ubyte")
    test_images = _read_encoded_images("polar_mnist.data", "t10k-images-idx3-ubyte")
    test_labels = _read_encoded_labels("polar_mnist.data", "t10k-labels-idx1-ubyte")
    return train_images, train_labels, test_images, test_labels


class _MNISTProjectorDataset(Dataset):
    def __init__(self, images: NDArray, labels: NDArray) -> None:
        self.images = images
        self.labels = labels

        # Pre-calculated from the training set at 128 resolution
        MEAN = 6005.8345
        STD = 3537.1011

        self.transform = transforms.Compose(
            [
                PolarAggregation(resolution=128),
                ToTensor1D(),
                Normalize1D(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return (self.transform(self.images[idx]), int(self.labels[idx]))


def get_train_loader(batch_size: int, shuffle: bool = True) -> DataLoader:
    train_images, train_labels, _, _ = load_mnist()
    dataset = _MNISTProjectorDataset(train_images, train_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_test_loader(batch_size: int) -> DataLoader:
    _, _, test_images, test_labels = load_mnist()
    dataset = _MNISTProjectorDataset(test_images, test_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
