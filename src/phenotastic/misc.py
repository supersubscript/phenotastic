import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cut(img: NDArray[np.floating[Any]], cuts: NDArray[np.integer[Any]]) -> NDArray[np.floating[Any]]:
    """Slice 3D image using specified cut coordinates.

    Args:
        img: 3D image array
        cuts: 3x2 array defining [start, end] for each dimension

    Returns:
        Sliced image array
    """
    # TODO this should be written for nD with advanced slicing comprehension
    return img[cuts[0, 0] : cuts[0, 1], cuts[1, 0] : cuts[1, 1], cuts[2, 0] : cuts[2, 1]]


def merge(lists: list[list[Any]]) -> list[set[Any]]:
    """Merge lists that share common elements.

    Groups lists that have any overlapping elements into single sets.

    Args:
        lists: List of lists to merge

    Returns:
        Minimal list of independent sets
    """
    sets = [set(lst) for lst in lists if lst]
    merged = 1
    while merged:
        merged = 0
        results: list[set[Any]] = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = 1
                    common |= x
            results.append(common)
        sets = results
    return sets


def flatten(arr: list[list[Any]]) -> list[Any]:
    """Flatten nested list into single-level list.

    Args:
        arr: Nested list

    Returns:
        Flattened list
    """
    import itertools

    return list(itertools.chain.from_iterable(arr))


def remove_empty_slices(arr: NDArray[Any], keepaxis: int = 0) -> NDArray[Any]:
    """Remove slices with no signal along specified axis.

    Args:
        arr: N-dimensional array
        keepaxis: Axis to preserve (remove slices along other axes)

    Returns:
        Array with empty slices removed
    """
    not_empty = np.sum(arr, axis=tuple(np.delete(list(range(arr.ndim)), keepaxis))) > 0
    arr = arr[not_empty]
    return arr


def reject_outliers(data: NDArray[np.floating[Any]], n: float = 2.0) -> NDArray[np.floating[Any]]:
    """Remove outliers outside of n standard deviations.

    Args:
        data: 1D array containing data to be filtered.
        n: Number of standard deviations that should be included in final data.

    Returns:
        Data within the specified range.
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    filtered_data = data[s < n]
    return filtered_data


def angle(v1: ArrayLike, v2: ArrayLike, acute: bool = False) -> float:
    """Compute angle between two vectors.

    Args:
        v1: First vector
        v2: Second vector
        acute: If True, return acute angle; otherwise return full angle

    Returns:
        Angle in radians
    """
    v1_arr = np.asarray(v1)
    v2_arr = np.asarray(v2)
    ang = float(np.arccos(np.dot(v1_arr, v2_arr) / (np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr))))
    return ang if acute else float(2 * np.pi - ang)


def angle_difference(ang1: ArrayLike, ang2: ArrayLike, period: float = 360) -> NDArray[np.floating[Any]]:
    """Compute smallest angular difference with periodic boundary conditions.

    Args:
        ang1: First angle(s) in degrees
        ang2: Second angle(s) in degrees
        period: Period of angular space (default 360 degrees)

    Returns:
        Smallest angular difference(s) in degrees
    """
    difference = np.subtract(ang1, ang2)
    angs = np.array([np.abs(np.mod(difference, period)), np.abs(np.mod(difference, -period))])
    angs = np.min(angs, axis=0)
    return angs


def divergence_angles(angles: ArrayLike, period: float = 360) -> NDArray[np.floating[Any]]:
    """Compute divergence angles between consecutive angles.

    Args:
        angles: Ordered array of angles in degrees
        period: Period of angular space (default 360 degrees)

    Returns:
        Array of divergence angles between consecutive pairs
    """
    angles_arr = np.asarray(angles)
    div_angs = angle_difference(angles_arr, np.roll(angles_arr, 1), period=period)[1:]

    return div_angs


def paraboloid(x: float, y: float, p: Sequence[float]) -> float:
    """Evaluate paraboloid z-value at given x, y coordinates.

    Paraboloid equation: z = p0*x² + p1*y² + p2*x + p3*y + p4

    Args:
        x: X coordinate
        y: Y coordinate
        p: 5-element array [p0, p1, p2, p3, p4] of paraboloid coefficients

    Returns:
        Z coordinate value
    """
    p1, p2, p3, p4, p5 = p
    z = p1 * x**2 + p2 * y**2 + p3 * x + p4 * y + p5
    return z


def get_max_contrast_colours(n: int = 64) -> list[list[int]]:
    """Get colors with maximal inter-color contrast.

    Args:
        n: Numbers of RGB colors to return.

    Returns:
        List of colours (RGB) up to a certain n that maximise contrast.
    """
    rgbs = [
        [0, 0, 0],
        [1, 0, 103],
        [213, 255, 0],
        [255, 0, 86],
        [158, 0, 142],
        [14, 76, 161],
        [255, 229, 2],
        [0, 95, 57],
        [0, 255, 0],
        [149, 0, 58],
        [255, 147, 126],
        [164, 36, 0],
        [0, 21, 68],
        [145, 208, 203],
        [98, 14, 0],
        [107, 104, 130],
        [0, 0, 255],
        [0, 125, 181],
        [106, 130, 108],
        [0, 174, 126],
        [194, 140, 159],
        [190, 153, 112],
        [0, 143, 156],
        [95, 173, 78],
        [255, 0, 0],
        [255, 0, 246],
        [255, 2, 157],
        [104, 61, 59],
        [255, 116, 163],
        [150, 138, 232],
        [152, 255, 82],
        [167, 87, 64],
        [1, 255, 254],
        [255, 238, 232],
        [254, 137, 0],
        [189, 198, 255],
        [1, 208, 255],
        [187, 136, 0],
        [117, 68, 177],
        [165, 255, 210],
        [255, 166, 254],
        [119, 77, 0],
        [122, 71, 130],
        [38, 52, 0],
        [0, 71, 84],
        [67, 0, 44],
        [181, 0, 255],
        [255, 177, 103],
        [255, 219, 102],
        [144, 251, 146],
        [126, 45, 210],
        [189, 211, 147],
        [229, 111, 254],
        [222, 255, 116],
        [0, 255, 120],
        [0, 155, 255],
        [0, 100, 1],
        [0, 118, 255],
        [133, 169, 0],
        [0, 185, 23],
        [120, 130, 49],
        [0, 255, 198],
        [255, 110, 65],
        [232, 94, 190],
    ]
    return rgbs[:n]


def prime_sieve(n: int, output: dict[int, bool] | list[int] | None = None) -> dict[int, bool] | list[int] | None:
    """Return a dict or a list of primes up to N create full prime sieve for
    N=10^6 in 1 sec."""
    if output is None:
        output = {}
    nroot = int(math.sqrt(n))
    sieve = list(range(n + 1))
    sieve[1] = 0

    for i in range(2, nroot + 1):
        if sieve[i] != 0:
            m = n / i - i
            sieve[i * i : n + 1 : i] = [0] * int(m + 1)

    if isinstance(output, dict):
        pmap: dict[int, bool] = {}
        for x in sieve:
            if x != 0:
                pmap[x] = True
        return pmap
    elif isinstance(output, list):
        return [x for x in sieve if x != 0]
    else:
        return None


def get_factors(n: int, primelist: list[int] | None = None) -> list[int]:
    """Get a list of all factors for N.

    Example:
        >>> get_factors(10)
        [1, 2, 5, 10]
    """
    if primelist is None:
        result = prime_sieve(n, output=[])
        primelist = result if isinstance(result, list) else []

    fcount: dict[int, int] = {}
    n_float = float(n)
    for p in primelist:
        if p > n_float:
            break
        if n_float % p == 0:
            fcount[p] = 0

        while n_float % p == 0:
            n_float /= p
            fcount[p] += 1

    factors = [1]
    for i in fcount:
        level = []
        exp = [i ** (x + 1) for x in range(fcount[i])]
        for j in exp:
            level.extend([j * x for x in factors])
        factors.extend(level)

    return factors


def get_prime_factors(n: int, primelist: list[int] | None = None) -> list[tuple[int, int]]:
    """Get a list of prime factors and corresponding powers.

    Example:
        >>> get_prime_factors(140)  # 140 = 2^2 * 5^1 * 7^1
        [(2, 2), (5, 1), (7, 1)]
    """
    if primelist is None:
        result = prime_sieve(n, output=[])
        primelist = result if isinstance(result, list) else []

    fs: list[tuple[int, int]] = []
    n_float = float(n)
    for p in primelist:
        count = 0
        while n_float % p == 0:
            n_float /= p
            count += 1
        if count > 0:
            fs.append((p, count))

    return fs


def coord_array(
    arr: NDArray[Any],
    res: tuple[float, float, float] = (1, 1, 1),
    offset: tuple[float, float, float] = (0, 0, 0),
) -> NDArray[np.floating[Any]]:
    """Create coordinate array matching dimensions of input array.

    Generates 3D coordinate grid with specified resolution and offset.

    Args:
        arr: 3D array defining output dimensions
        res: Spatial resolution in XYZ dimensions
        offset: Origin offset in XYZ

    Returns:
        3xN array of XYZ coordinates (N = arr.size)
    """
    xv = offset[0] + np.arange(0, arr.shape[0] * res[0] - 0.000001, res[0])
    yv = offset[1] + np.arange(0, arr.shape[1] * res[1] - 0.000001, res[1])
    zv = offset[2] + np.arange(0, arr.shape[2] * res[2] - 0.000001, res[2])
    grid_list: list[NDArray[np.floating[Any]]] = list(np.meshgrid(xv, yv, zv))
    grid_arr: NDArray[np.floating[Any]] = np.array(grid_list)
    grid_arr = grid_arr.transpose(0, 2, 1, 3)
    xx, yy, zz = grid_arr
    xx = xx.ravel()
    yy = yy.ravel()
    zz = zz.ravel()

    # Make compatible lists
    coords = np.vstack((xx.ravel(), yy.ravel(), zz.ravel()))
    return coords


def rot_matrix_44(angles: Sequence[float], invert: bool = False) -> NDArray[np.floating[Any]]:
    """Generate 4x4 homogeneous rotation matrix.

    Args:
        angles: Rotation angles [alpha, beta, gamma] in radians
        invert: If True, return inverse rotation matrix

    Returns:
        4x4 rotation matrix
    """
    alpha, beta, gamma = angles
    rot_x = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(beta), 0, np.sin(beta), 0],
            [0, 1, 0, 0],
            [-np.sin(beta), 0, np.cos(beta), 0],
            [0, 0, 0, 1],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0, 0],
            [np.sin(gamma), np.cos(gamma), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    if invert:
        rot_matrix = np.linalg.inv(np.matmul(np.matmul(rot_z, rot_y), rot_x))
    else:
        rot_matrix = np.matmul(np.matmul(rot_z, rot_y), rot_x)

    return rot_matrix


def rotate(
    coord: NDArray[np.floating[Any]], angles: Sequence[float], invert: bool = False
) -> NDArray[np.floating[Any]]:
    """Rotate coordinates by Euler angles.

    Applies rotations around x, y, z axes in order beta-gamma-alpha.

    Args:
        coord: Nx3 array of coordinates
        angles: Rotation angles [alpha, beta, gamma] in radians
        invert: If True, apply inverse rotation

    Returns:
        Rotated coordinates as Nx3 array
    """
    alpha, beta, gamma = angles
    xyz = np.zeros(np.shape(coord))
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )
    rot_y = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    rot_z = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    rot_matrix = (
        np.linalg.inv(np.matmul(np.matmul(rot_z, rot_y), rot_x))
        if invert
        else np.matmul(np.matmul(rot_z, rot_y), rot_x)
    )

    for ii in range(np.shape(coord)[0]):
        xyz[ii, :] = rot_matrix.dot(np.array(coord[ii, :]))
    return xyz


def match_shape(
    a: NDArray[Any],
    t: Sequence[int],
    side: str = "both",
    val: int | float | complex | str = 0,
) -> NDArray[Any]:
    """Pad or trim array to match target shape.

    Args:
        a: Input array
        t: Dimensions to pad/trim to, must be a list or tuple
        side: One of 'both', 'before', and 'after'
        val: Value to pad with (numeric or 'max', 'mean', 'median', 'min')

    Returns:
        The padded/trimmed array
    """
    if len(t) != a.ndim:
        raise ValueError("t shape must have the same number of dimensions as the input")

    if isinstance(val, (int, float, complex)):
        b = np.ones(t, a.dtype) * val
    elif val == "max":
        b = np.ones(t, a.dtype) * np.max(a)
    elif val == "mean":
        b = np.ones(t, a.dtype) * np.mean(a)
    elif val == "median":
        b = np.ones(t, a.dtype) * np.median(a)
    elif val == "min":
        b = np.ones(t, a.dtype) * np.min(a)
    else:
        raise ValueError(f"Invalid pad value: {val}")

    aind: list[slice] = [slice(None, None)] * a.ndim
    bind: list[slice] = [slice(None, None)] * a.ndim

    # pad/trim comes after the array in each dimension
    if side == "after":
        for dd in range(a.ndim):
            if a.shape[dd] > t[dd]:
                aind[dd] = slice(None, t[dd])
            elif a.shape[dd] < t[dd]:
                bind[dd] = slice(None, a.shape[dd])
    # pad/trim comes before the array in each dimension
    elif side == "before":
        for dd in range(a.ndim):
            if a.shape[dd] > t[dd]:
                aind[dd] = slice(int(a.shape[dd] - t[dd]), None)
            elif a.shape[dd] < t[dd]:
                bind[dd] = slice(int(t[dd] - a.shape[dd]), None)
    # pad/trim both sides of the array in each dimension
    elif side == "both":
        for dd in range(a.ndim):
            if a.shape[dd] > t[dd]:
                diff = (a.shape[dd] - t[dd]) / 2.0
                aind[dd] = slice(int(np.floor(diff)), int(a.shape[dd] - np.ceil(diff)))
            elif a.shape[dd] < t[dd]:
                diff = (t[dd] - a.shape[dd]) / 2.0
                bind[dd] = slice(int(np.floor(diff)), int(t[dd] - np.ceil(diff)))
    else:
        raise ValueError(f"Invalid choice of pad type: {side}")

    b[tuple(bind)] = a[tuple(aind)]

    return b


def mode(x: Sequence[Any]) -> Any:
    """Compute the mode of a sequence of values.

    Returns nan if sequence too short.
    """
    if len(x) < 1:
        return np.nan
    return max(list(x), key=list(x).count)


def car2sph(xyz: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Convert Cartesian to spherical coordinates.

    Converts (x, y, z) to (r, theta, phi) where theta=0 along z-axis.

    Args:
        xyz: Nx3 array of Cartesian coordinates

    Returns:
        Nx3 array of spherical coordinates [r, theta, phi]
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    rtp = np.zeros(xyz.shape)
    xy = x**2 + y**2
    rtp[:, 0] = np.sqrt(xy + z**2)
    rtp[:, 1] = np.arctan2(np.sqrt(xy), z)
    rtp[:, 2] = np.arctan2(y, x)
    return rtp


def sph2car(rtp: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Convert spherical to Cartesian coordinates.

    Converts (r, theta, phi) to (x, y, z) where theta=0 along z-axis.

    Args:
        rtp: Nx3 array of spherical coordinates [r, theta, phi]

    Returns:
        Nx3 array of Cartesian coordinates [x, y, z]
    """
    xyz = np.zeros(rtp.shape)
    xyz[:, 0] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
    xyz[:, 1] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
    xyz[:, 2] = rtp[:, 0] * np.cos(rtp[:, 1])
    return xyz


def to_uint8(data: NDArray, normalize: bool = True) -> NDArray[np.uint8]:
    """Convert image array to uint8 format.

    Args:
        data: Input image array
        normalize: If True, normalize to [0, 255]; if False, scale proportionally

    Returns:
        uint8 image array
    """
    data_float = data.astype(float)
    if normalize:
        data_float = (data_float - np.min(data_float)) / (np.max(data_float) - np.min(data_float)) * 255
    else:
        data_float = data_float / np.max(data_float) * 255
    return data_float.astype(np.uint8)


def matching_rows(array1: NDArray[Any], array2: NDArray[Any]) -> NDArray[np.intp]:
    """Find matching rows in a 2D array."""
    return np.array(np.all((array1[:, None, :] == array2[None, :, :]), axis=-1).nonzero()).T


def rand_cmap(
    nlabels: int,
    cmap_type: str = "bright",
    first_color_black: bool = True,
    last_color_black: bool = False,
    verbose: bool = False,
) -> Any:
    """Creates a random colormap to be used together with matplotlib.

    Useful for segmentation tasks.

    Args:
        nlabels: Number of labels (size of colormap)
        cmap_type: 'bright' for strong colors, 'soft' for pastel colors.
        first_color_black: Option to use first color as black.
        last_color_black: Option to use last color as black.
        verbose: Prints the number of labels and shows the colormap.

    Returns:
        Colormap for matplotlib
    """
    import colorsys

    from matplotlib.colors import LinearSegmentedColormap

    if cmap_type not in ("bright", "soft"):
        print('Please choose "bright" or "soft" for type')
        return None

    if verbose:
        print("Number of labels: " + str(nlabels))

    rand_rgb_colors: list[list[float] | tuple[float, ...]] = []

    # Generate color map for bright colors, based on hsv
    if cmap_type == "bright":
        rand_hsv_colors = [
            (
                np.random.uniform(low=0.0, high=1),
                np.random.uniform(low=0.2, high=1),
                np.random.uniform(low=0.9, high=1),
            )
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        for hsv_color in rand_hsv_colors:
            rand_rgb_colors.append(colorsys.hsv_to_rgb(hsv_color[0], hsv_color[1], hsv_color[2]))

        if first_color_black:
            rand_rgb_colors[0] = [0, 0, 0]

        if last_color_black:
            rand_rgb_colors[-1] = [0, 0, 0]

    # Generate soft pastel colors, by limiting the RGB spectrum
    if cmap_type == "soft":
        low = 0.6
        high = 0.95
        rand_rgb_colors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

        if first_color_black:
            rand_rgb_colors[0] = [0, 0, 0]

        if last_color_black:
            rand_rgb_colors[-1] = [0, 0, 0]

    random_colormap = LinearSegmentedColormap.from_list("new_map", rand_rgb_colors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colorbar, colors
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        _ = colorbar.ColorbarBase(
            ax,
            cmap=random_colormap,
            norm=norm,
            spacing="proportional",
            ticks=None,
            boundaries=bounds,
            format="%1i",
            orientation="horizontal",
        )

    return random_colormap
