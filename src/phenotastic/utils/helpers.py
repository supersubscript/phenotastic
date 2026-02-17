"""Helper utility functions for phenotastic."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from lxml import objectify as xml_objectify
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable


@overload
def autocrop(
    array: NDArray[np.number],
    threshold: float,
    channel: int | list[int] | tuple[int, ...] | NDArray[np.integer],
    min_pixels_above_threshold: int,
    return_crop_bounds: Literal[False],
    padding: int | NDArray[np.integer] | None,
) -> NDArray[np.number]: ...


@overload
def autocrop(
    array: NDArray[np.number],
    threshold: float = ...,
    channel: int | list[int] | tuple[int, ...] | NDArray[np.integer] = ...,
    min_pixels_above_threshold: int = ...,
    return_crop_bounds: bool = ...,
    padding: int | NDArray[np.integer] | None = ...,
) -> NDArray[np.number] | tuple[NDArray[np.number], NDArray[np.integer]]: ...


def autocrop(
    array: NDArray[np.number],
    threshold: float = 8e3,
    channel: int | list[int] | tuple[int, ...] | NDArray[np.integer] = -1,
    min_pixels_above_threshold: int = 1,
    return_crop_bounds: bool = False,
    padding: int | NDArray[np.integer] | None = None,
) -> NDArray[np.number] | tuple[NDArray[np.number], NDArray[np.integer]]:
    """Automatically crop a 3D or 4D array to remove empty border regions.

    Finds the bounding box of regions where pixel values exceed the threshold
    and crops the array to that region, optionally with padding.

    Args:
        array: Input array with shape (Z, Y, X) for 3D or (Z, C, Y, X) for 4D.
        threshold: Pixel intensity threshold for detecting content.
        channel: For 4D arrays, which channel(s) to use for detection.
            -1 means use max across all channels.
        min_pixels_above_threshold: Minimum number of pixels above threshold
            required to consider a slice as containing content.
        return_crop_bounds: If True, also return the crop boundaries.
        padding: Extra pixels to include around the detected region.
            Can be a single int (same padding all sides) or array of shape (3, 2)
            for per-axis start/end padding.

    Returns:
        Cropped array, or tuple of (cropped_array, crop_bounds) if return_crop_bounds=True.
        crop_bounds has shape (3, 2) with [start, end] indices for each axis.
    """
    # Normalize padding to shape (3, 2)
    padding_array: NDArray[np.integer]
    if padding is None:
        padding_array = np.zeros((3, 2), dtype=np.int64)
    elif isinstance(padding, int):
        padding_array = np.full((3, 2), padding, dtype=np.int64)
    else:
        padding_array = np.asarray(padding, dtype=np.int64)

    # For 4D arrays, reduce to 3D by combining channels
    detection_array = _reduce_channels(array, channel) if array.ndim > 3 else array

    crop_bounds = _find_crop_bounds(detection_array, threshold, min_pixels_above_threshold)
    crop_bounds = _apply_padding(crop_bounds, padding_array, array.shape)
    cropped = _crop_array(array, crop_bounds)

    if return_crop_bounds:
        return cropped, crop_bounds
    return cropped


def _reduce_channels(
    array: NDArray[np.number],
    channel: int | list[int] | tuple[int, ...] | NDArray[np.integer],
) -> NDArray[np.number]:
    """Reduce a 4D array to 3D by combining channels."""
    if channel == -1:
        # Max across all channels
        return np.max(array, axis=1)
    elif isinstance(channel, (list, tuple, np.ndarray)):
        # Max across selected channels
        return np.max(np.take(array, channel, axis=1), axis=1)
    else:
        # Single channel
        return array[:, channel]


def _find_crop_bounds(
    array: NDArray[np.number],
    threshold: float,
    min_pixels: int,
) -> NDArray[np.integer]:
    """Find the crop boundaries for each axis based on threshold detection."""
    crop_bounds = np.zeros((array.ndim, 2), dtype=int)

    for axis in range(array.ndim):
        # Move target axis to front
        other_axes = tuple(i for i in range(array.ndim) if i != axis)
        transposed = np.transpose(array, (axis,) + other_axes)

        # Count pixels above threshold for each slice along this axis
        flattened = transposed.reshape(transposed.shape[0], -1)
        pixels_above_threshold = np.sum(flattened > threshold, axis=1)

        # Find first and last slices with enough pixels
        first_slice = _find_first_above(pixels_above_threshold, min_pixels)
        last_slice = len(pixels_above_threshold) - _find_first_above(pixels_above_threshold[::-1], min_pixels)

        crop_bounds[axis] = [first_slice, last_slice]

    return crop_bounds


def _find_first_above(values: NDArray[np.integer], threshold: int) -> int:
    """Find index of first value >= threshold, or 0 if none found."""
    indices = np.where(values >= threshold)[0]
    return int(indices[0]) if len(indices) > 0 else 0


def _apply_padding(
    crop_bounds: NDArray[np.integer],
    padding: NDArray[np.integer],
    array_shape: tuple[int, ...],
) -> NDArray[np.integer]:
    """Apply padding to crop bounds, clamping to array boundaries."""
    padded = crop_bounds.copy()

    for axis in range(min(3, len(array_shape))):
        # Subtract padding from start, add to end
        padded[axis, 0] = max(0, crop_bounds[axis, 0] - padding[axis, 0])
        padded[axis, 1] = min(array_shape[axis], crop_bounds[axis, 1] + padding[axis, 1])

    return padded


def _crop_array(
    array: NDArray[np.number],
    crop_bounds: NDArray[np.integer],
) -> NDArray[np.number]:
    """Crop array using the specified bounds for each axis."""
    # For 4D arrays, temporarily move channel axis to end
    if array.ndim > 3:
        array = np.moveaxis(array, 1, -1)

    # Crop each spatial axis
    for axis, (start, end) in enumerate(crop_bounds):
        array = np.swapaxes(array, 0, axis)
        array = array[start:end]
        array = np.swapaxes(array, 0, axis)

    # Restore channel axis position for 4D arrays
    if array.ndim > 3:
        array = np.moveaxis(array, -1, 1)

    return array


Resolution = tuple[float, float, float]  # type alias for voxel resolution
DEFAULT_RESOLUTION: Resolution = (1.0, 1.0, 1.0)


def _xml_to_dict(xml_object: object) -> Any:
    """Recursively convert lxml objectified XML to a nested dictionary."""
    dict_object = xml_object.__dict__
    if not dict_object:
        return xml_object
    for key, value in dict_object.items():
        dict_object[key] = _xml_to_dict(value)
    return dict_object


def _get_resolution_czi(filepath: str | Path) -> Resolution | None:
    """Extract resolution from CZI (Zeiss) file metadata."""
    try:
        import czifile as czi

        with czi.CziFile(filepath) as f:
            metadata = _xml_to_dict(xml_objectify.fromstring(f.metadata()))
            acq_meta = metadata["Metadata"]["Experiment"]["ExperimentBlocks"]["AcquisitionBlock"][
                "AcquisitionModeSetup"
            ]
            return (
                float(acq_meta.get("ScalingZ", 1.0)),
                float(acq_meta.get("ScalingY", 1.0)),
                float(acq_meta.get("ScalingX", 1.0)),
            )
    except (ImportError, KeyError, AttributeError, TypeError):
        return None


def _get_resolution_lif(filepath: str | Path) -> Resolution | None:
    """Extract resolution from LIF (Leica) file metadata."""
    try:
        import read_lif as lif

        reader = lif.Reader(filepath)
        series = reader.getSeries()
        if not series:
            return None
        metadata = series[0].getMetadata()
        return (
            float(metadata.get("voxel_size_z", 1.0)),
            float(metadata.get("voxel_size_y", 1.0)),
            float(metadata.get("voxel_size_x", 1.0)),
        )
    except (ImportError, IndexError, KeyError, AttributeError, TypeError):
        return None


def _get_resolution_tiff(filepath: str | Path) -> Resolution | None:
    """Extract resolution from TIFF file metadata (ImageJ or LSM)."""
    try:
        import tifffile as tiff

        with tiff.TiffFile(filepath) as f:
            if f.imagej_metadata is not None:
                z = float(f.imagej_metadata.get("spacing", 1.0))
                x_res_tag = f.pages[0].tags.get("XResolution")
                x = x_res_tag.value[1] / x_res_tag.value[0] if x_res_tag and x_res_tag.value[0] != 0 else 1.0
                return (z, float(x), float(x))

            if f.lsm_metadata is not None:
                return (
                    float(f.lsm_metadata.get("VoxelSizeZ", 1.0)),
                    float(f.lsm_metadata.get("VoxelSizeY", 1.0)),
                    float(f.lsm_metadata.get("VoxelSizeX", 1.0)),
                )
    except (ImportError, KeyError, AttributeError, TypeError, ZeroDivisionError):
        pass
    return None


def get_resolution(filepath: Path | str) -> Resolution:
    """Extract voxel resolution from image file metadata.

    Supports CZI (Zeiss), LIF (Leica), and TIFF (ImageJ/LSM) formats.
    Returns default resolution (1.0, 1.0, 1.0) if metadata cannot be extracted.

    Args:
        filepath: Path to image file

    Returns:
        Tuple of (z, y, x) resolution values in micrometers
    """
    ext = Path(filepath).suffix.lower()

    # Format-specific handlers
    handlers: dict[str, Callable[[str | Path], Resolution | None]] = {
        ".czi": _get_resolution_czi,
        ".lif": _get_resolution_lif,
    }

    if ext in handlers:
        result = handlers[ext](filepath)
        return result if result is not None else DEFAULT_RESOLUTION

    # Default to TIFF handling for other extensions
    result = _get_resolution_tiff(filepath)
    return result if result is not None else DEFAULT_RESOLUTION
