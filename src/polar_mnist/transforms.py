from __future__ import annotations

from numpy.typing import NDArray
import torch
import cv2


class PolarAggregation:
    """
    Calculate a 1D radial distribution signal from the center of a 2D image.
    """

    def __init__(self, resolution: int):
        self.resolution = resolution

        half_size = resolution // 2
        self._origin = (half_size, half_size)
        self._max_radius = (half_size**2.0 + half_size**2.0) ** 0.5

    def __call__(self, image: NDArray) -> NDArray:
        resampled = cv2.resize(image, (self.resolution, self.resolution))
        polar = cv2.linearPolar(
            resampled,
            self._origin,
            self._max_radius,
            cv2.WARP_FILL_OUTLIERS,
        )

        return polar.sum(axis=1)


class Normalize1D:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ToTensor1D:
    def __call__(self, x: NDArray) -> torch.Tensor:
        return torch.from_numpy(x).float().unsqueeze(0)
