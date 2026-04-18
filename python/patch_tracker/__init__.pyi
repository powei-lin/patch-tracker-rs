from typing import overload
import numpy as np
from numpy.typing import NDArray

class ImagePyramid:
    """A sequence of grayscale images at successive half-resolution levels."""

    def __len__(self) -> int:
        """Return the number of pyramid levels."""
        ...

    def __getitem__(self, index: int) -> NDArray[np.uint8]:
        """Return the image at *index* as a ``(H, W)`` uint8 numpy array."""
        ...

    def __repr__(self) -> str: ...

def build_image_pyramid(image: NDArray[np.uint8], levels: int) -> ImagePyramid:
    """Build an image pyramid from a grayscale image.

    Args:
        image: Grayscale image, shape ``(H, W)`` or ``(H, W, 1)``, dtype ``uint8``.
        levels: Number of pyramid levels.

    Returns:
        :class:`ImagePyramid` with ``levels`` images, each half the resolution of the previous.
    """
    ...

def track_points(
    pyramid0: ImagePyramid,
    pyramid1: ImagePyramid,
    points: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Track points from one image pyramid to another.

    This is a lower-level function. For typical use, prefer :class:`PatchTracker`.

    Args:
        pyramid0: Source image pyramid (previous frame).
        pyramid1: Target image pyramid (current frame).
        points: Dict mapping track id to ``(x, y)`` coordinates in *pyramid0*.

    Returns:
        Dict mapping surviving track ids to new ``(x, y)`` coordinates in *pyramid1*.
    """
    ...

class PatchTracker:
    """Single-camera optical flow patch tracker."""

    def __init__(self, levels: int = 4, grid_size: int = 20) -> None:
        """Create a new PatchTracker.

        Args:
            levels: Number of image pyramid levels. Default 4.
            grid_size: Grid cell size in pixels for keypoint distribution. Default 20.
        """
        ...

    def process_frame(self, image: NDArray[np.uint8]) -> None:
        """Process a new frame. Tracks existing points and detects new ones.

        Args:
            image: Grayscale image as numpy array, shape ``(H, W)`` or ``(H, W, 1)``,
                dtype ``uint8``.
        """
        ...

    def get_track_points(self) -> dict[int, tuple[float, float]]:
        """Return currently tracked points.

        Returns:
            Dict mapping track id to ``(x, y)`` pixel coordinates.
        """
        ...

    def remove_id(self, ids: list[int]) -> None:
        """Remove tracked points by id.

        Args:
            ids: List of track ids to remove.
        """
        ...

    def add_points(self, points: list[tuple[float, float]]) -> None:
        """Manually add points to the tracker.

        Args:
            points: List of ``(x, y)`` pixel coordinates.
        """
        ...

class StereoPatchTracker:
    """Stereo-camera optical flow patch tracker."""

    def __init__(self, levels: int = 4, grid_size: int = 20) -> None:
        """Create a new StereoPatchTracker.

        Args:
            levels: Number of image pyramid levels. Default 4.
            grid_size: Grid cell size in pixels for keypoint distribution. Default 20.
        """
        ...

    def process_frame(
        self,
        image0: NDArray[np.uint8],
        image1: NDArray[np.uint8],
    ) -> None:
        """Process a new stereo frame pair.

        Args:
            image0: Left grayscale image, shape ``(H, W)`` or ``(H, W, 1)``, dtype ``uint8``.
            image1: Right grayscale image, shape ``(H, W)`` or ``(H, W, 1)``, dtype ``uint8``.
        """
        ...

    def get_track_points(
        self,
    ) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
        """Return currently tracked points for both cameras.

        Returns:
            Tuple of two dicts, each mapping track id to ``(x, y)`` coordinates.
            The same id in both dicts refers to the same stereo-matched feature.
        """
        ...

    def remove_id(self, ids: list[int]) -> None:
        """Remove tracked points by id.

        Args:
            ids: List of track ids to remove.
        """
        ...
