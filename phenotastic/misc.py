#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:00:11 2018

@author: henrik
"""

import math
import os
import re

import numpy as np


def cut(img, cuts):
    return img[
        cuts[0, 0]: cuts[0, 1], cuts[1, 0]: cuts[1, 1], cuts[2, 0]: cuts[2, 1]
    ]


def merge(lists):
    """
    Merge lists based on overlapping elements.

    Parameters
    ----------
    lists : list of lists
        Lists to be merged.

    Returns
    -------
    sets : list
        Minimal list of independent sets.

    """
    sets = [set(lst) for lst in lists if lst]
    merged = 1
    while merged:
        merged = 0
        results = []
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


def flatten(llist):
    """Flatten a list of lists"""
    return [item for sublist in llist for item in sublist]


def remove_empty_slices(arr, keepaxis=0):
    """Remove empty slices (based on the total intensity) in an ndarray"""
    not_empty = np.sum(arr, axis=tuple(
        np.delete(list(range(arr.ndim)), keepaxis))) > 0
    arr = arr[not_empty]
    return arr


def reject_outliers(data, n=2.0):
    """Remove outliers outside of n standard deviations.

    Parameters
    ----------
    data : np.array
        1D array containing data to be filtered.

    n : float
         Number of standard deviations that should be included in final data.
         (Default = 2.)

    Returns
    -------
    filtered_data : bool, np.array
        Data within the specified range.

    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    filtered_data = data[s < n]
    return filtered_data


def angle(v1, v2, acute=False):
    angle = np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle if acute else 2 * np.pi - angle


def angle_difference(ang1, ang2, period=360):
    """Return the smallest angle difference on a template with periodic boundary conditions.

    Returns
    -------
    angs : np.array of floats
        The smallest angles.

    """
    difference = np.subtract(ang1, ang2)
    angs = np.array(
        [np.abs(np.mod(difference, period)),
         np.abs(np.mod(difference, -period))]
    )
    angs = np.min(angs, axis=0)
    return angs


def divergence_angles(angles, period=360):
    """Divergence angles between an ordered list of angles"""
    div_angs = angle_difference(angles, np.roll(angles, 1), period=period)[1:]

    return div_angs


def paraboloid(x, y, p):
    """Return the z-value for a paraboloid given input xy-coordinates and
    parameters.

    Parameters
    ----------
    x : float
        x-coordinate.
    y : float
        y-coordinate.
    p : 5-length np.array
        Paraboloid parameters.

    Returns
    -------

    """
    p1, p2, p3, p4, p5 = p
    return p1 * x**2 + p2 * y**2 + p3 * x + p4 * y + p5


def get_max_contrast_colours(n=64):
    """Get colors with maximal inter-color contrast.

    Parameters
    ----------
    n : Numbers of RGB colors to return.
         (Default value = 64)

    Returns
    -------
    rgbs : list
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
    return rgbs[0:n]


def prime_sieve(n, output={}):
    """
    Return a dict or a list of primes up to N create full prime sieve for
    N=10^6 in 1 sec

    """

    nroot = int(math.sqrt(n))
    sieve = list(range(n + 1))
    sieve[1] = 0

    for i in range(2, nroot + 1):
        if sieve[i] != 0:
            m = n / i - i
            sieve[i * i: n + 1: i] = [0] * (m + 1)

    if type(output) == dict:
        pmap = {}
        for x in sieve:
            if x != 0:
                pmap[x] = True
        return pmap
    elif type(output) == list:
        return [x for x in sieve if x != 0]
    else:
        return None


def get_factors(n, primelist=None):
    """
    Get a list of all factors for N.

    Example
    -------
    >>> get_factors(10)
    >>> Out[1]: [1, 2, 5, 10]

    """
    if primelist is None:
        primelist = prime_sieve(n, output=[])

    fcount = {}
    for p in primelist:
        if p > n:
            break
        if n % p == 0:
            fcount[p] = 0

        while n % p == 0:
            n /= p
            fcount[p] += 1

    factors = [1]
    for i in fcount:
        level = []
        exp = [i ** (x + 1) for x in range(fcount[i])]
        for j in exp:
            level.extend([j * x for x in factors])
        factors.extend(level)

    return factors


def get_prime_factors(n, primelist=None):
    """Get a list of prime factors and corresponding powers.

    Example
    -------
    >>> get_prime_factors(140) # 140 = 2^2 * 5^1 * 7^1
    >>> Out[1]: [(2, 2), (5, 1), (7, 1)]

    """
    if primelist is None:
        primelist = prime_sieve(n, output=[])

    fs = []
    for p in primelist:
        count = 0
        while n % p == 0:
            n /= p
            count += 1
        if count > 0:
            fs.append((p, count))

    return fs


def autocrop(arr, threshold=8e3, channel=-1, n=1, return_cuts=False, offset=None):

    if offset is None:
        offset = [[0, 0], [0, 0], [0, 0]]
    offset = np.array(offset)

    sumarr = arr
    if arr.ndim > 3:
        if channel == -1:
            sumarr = np.max(arr, axis=1)
        elif isinstance(channel, (list, np.ndarray, tuple)):
            sumarr = np.max(arr.take(channel, axis=1), axis=1)
        else:
            sumarr = sumarr[:, channel]

    cp = np.zeros((sumarr.ndim, 2), dtype=np.int)
    for ii in range(sumarr.ndim):
        axes = np.array([0, 1, 2])[np.array([0, 1, 2]) != ii]

        transposed = np.transpose(sumarr, (ii,) + tuple(axes))
        nabove = np.sum(
            np.reshape(transposed, (transposed.shape[0], -1)) > threshold, axis=1
        )

        first = next((e[0] for e in enumerate(nabove) if e[1] >= n), 0)
        last = len(nabove) - next(
            (e[0] for e in enumerate(nabove[::-1]) if e[1] >= n), 0
        )

        cp[ii] = first, last
    #    ranges = [range(cp[ii, 0], cp[ii, 1]) for ii in range(len(cp))]
    cp[0, 0] = np.max([0, cp[0, 0] - offset[0, 0]])
    cp[0, 1] = np.min([arr.shape[0], cp[0, 1] + offset[0, 1]])
    cp[1, 0] = np.max([0, cp[1, 0] - offset[1, 0]])
    cp[1, 1] = np.min([arr.shape[1], cp[1, 1] + offset[1, 1]])
    cp[2, 0] = np.max([0, cp[2, 0] - offset[2, 0]])
    cp[2, 1] = np.min([arr.shape[2], cp[2, 1] + offset[2, 1]])

    if arr.ndim > 3:
        arr = np.moveaxis(arr, 1, -1)
    for ii, _range in enumerate(cp):
        arr = np.swapaxes(arr, 0, ii)
        arr = arr[_range[0]: _range[1]]
        arr = np.swapaxes(arr, 0, ii)
    if arr.ndim > 3:
        arr = np.moveaxis(arr, -1, 1)

    if return_cuts:
        return arr, cp
    else:
        return arr


def coord_array(arr, res=(1, 1, 1), offset=(0, 0, 0)):
    """
    Create a coordinate array (of e.g. same dimensionality as intensity array) of the same dimensions as another array. Only defined for 3D.

    Parameters
    ----------
    arr : np.ndarray
        Array defining dimensions of coordinate array
    res : 3-tuple, optional
        Resolution of the array in the three dimensions. Default = (1,1,1).
    offset : 3-tuple, optional
        Origin offset. Default = (0,0,0).

    Note
    ----
    Only 3D.

    Returns
    -------
    coords : np.ndarray
        Vertically stacked coordinate array for the input data.
    """

    xv = offset[0] + np.arange(0, arr.shape[0] * res[0] - 0.000001, res[0])
    yv = offset[1] + np.arange(0, arr.shape[1] * res[1] - 0.000001, res[1])
    zv = offset[2] + np.arange(0, arr.shape[2] * res[2] - 0.000001, res[2])
    grid = np.meshgrid(xv, yv, zv)
    grid = np.array(grid)
    grid = grid.transpose(0, 2, 1, 3)
    xx, yy, zz = grid
    xx = xx.ravel()
    yy = yy.ravel()
    zz = zz.ravel()

    # Make compatible lists
    coords = np.vstack((xx.ravel(), yy.ravel(), zz.ravel()))
    return coords


def rot_matrix_44(angles, invert=False):
    alpha, beta, gamma = angles
    Rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )
    Ry = np.array(
        [
            [np.cos(beta), 0, np.sin(beta), 0],
            [0, 1, 0, 0],
            [-np.sin(beta), 0, np.cos(beta), 0],
            [0, 0, 0, 1],
        ]
    )
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0, 0],
            [np.sin(gamma), np.cos(gamma), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    if invert == True:
        R = np.linalg.inv(np.matmul(np.matmul(Rz, Ry), Rx))
    elif invert == False:
        R = np.matmul(np.matmul(Rz, Ry), Rx)

    return R


def rotate(coord, angles, invert=False):
    """
    Rotate given coordinates by specified angles.

    Use rotation matrices to rotate a list of coordinates around the x, y, z axis
    by specified angles alpha, beta, gamma.

    """

    alpha, beta, gamma = angles
    xyz = np.zeros(np.shape(coord))
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0],
         [-np.sin(beta), 0, np.cos(beta)]]
    )
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    if invert:
        R = np.linalg.inv(np.matmul(np.matmul(Rz, Ry), Rx))
    else:
        R = np.matmul(np.matmul(Rz, Ry), Rx)

    for ii in range(np.shape(coord)[0]):
        xyz[ii, :] = R.dot(np.array(coord[ii, :]))
    return xyz


def mkdir(path):
    """
    Make a directory if it doesn't exist already.

    Parameters
    ----------
    path : str
        Path to directory.

    Returns
    -------
    None.

    """

    if not os.path.exists(path):
        os.makedirs(path)


def listdir(path, include=None, exclude=None, full=True, sorting=None):
    if isinstance(path, (list, np.ndarray)):
        files = flatten([listdir(ff, include, exclude, full, sorting)
                        for ff in path])
        return np.array(files)
    else:
        files = os.listdir(path)
        files = np.array(files)

    if full:
        files = np.array([os.path.join(path, x) for x in files])

    # Include
    if isinstance(include, str):
        files = np.array([x for x in files if include in x])
    elif isinstance(include, (list, np.ndarray)):
        matches = np.array([np.array([inc in ii for ii in files])
                           for inc in include])
        matches = np.any(matches, axis=0)
        files = files[matches]

    # Exclude
    if isinstance(exclude, str):
        files = np.array([x for x in files if exclude not in x])
    elif isinstance(exclude, (list, np.ndarray)):
        matches = np.array([np.array([exc in ii for ii in files])
                           for exc in exclude])
        matches = np.logical_not(np.any(matches, axis=0))
        files = files[matches]

    if sorting == 'natural':
        files = np.array(natural_sort(files))
    elif sorting == 'alphabetical':
        files = np.sort(files)
    elif sorting == True:
        files = np.sort(files)

    return files


def natural_sort(l):
    """
    Sort a list alphanumerically (natural sorting).

    Parameters
    ----------
    l : list or np.ndarray
        Structure to sort.

    Returns
    -------
    list or np.ndarray
        Sorted list/array.

    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def match_shape(a, t, side='both', val=0):
    """

    Parameters
    ----------
    a : np.ndarray
    t : Dimensions to pad/trim to, must be a list or tuple
    side : One of 'both', 'before', and 'after'
    val : value to pad with
    """
    try:
        if len(t) != a.ndim:
            raise TypeError(
                't shape must have the same number of dimensions as the input'
            )
    except TypeError:
        raise TypeError('t must be array-like')

    try:
        if isinstance(val, (int, float, complex)):
            b = np.ones(t, a.dtype) * val
        elif val == 'max':
            b = np.ones(t, a.dtype) * np.max(a)
        elif val == 'mean':
            b = np.ones(t, a.dtype) * np.mean(a)
        elif val == 'median':
            b = np.ones(t, a.dtype) * np.median(a)
        elif val == 'min':
            b = np.ones(t, a.dtype) * np.min(a)
    except TypeError:
        raise TypeError('Pad value must be numeric or string')
    except ValueError:
        raise ValueError('Pad value must be scalar or valid string')

    aind = [slice(None, None)] * a.ndim
    bind = [slice(None, None)] * a.ndim

    # pad/trim comes after the array in each dimension
    if side == 'after':
        for dd in range(a.ndim):
            if a.shape[dd] > t[dd]:
                aind[dd] = slice(None, t[dd])
            elif a.shape[dd] < t[dd]:
                bind[dd] = slice(None, a.shape[dd])
    # pad/trim comes before the array in each dimension
    elif side == 'before':
        for dd in range(a.ndim):
            if a.shape[dd] > t[dd]:
                aind[dd] = slice(int(a.shape[dd] - t[dd]), None)
            elif a.shape[dd] < t[dd]:
                bind[dd] = slice(int(t[dd] - a.shape[dd]), None)
    # pad/trim both sides of the array in each dimension
    elif side == 'both':
        for dd in range(a.ndim):
            if a.shape[dd] > t[dd]:
                diff = (a.shape[dd] - t[dd]) / 2.0
                aind[dd] = slice(int(np.floor(diff)), int(
                    a.shape[dd] - np.ceil(diff)))
            elif a.shape[dd] < t[dd]:
                diff = (t[dd] - a.shape[dd]) / 2.0
                bind[dd] = slice(int(np.floor(diff)),
                                 int(t[dd] - np.ceil(diff)))
    else:
        raise Exception('Invalid choice of pad type: %s' % side)

    b[tuple(bind)] = a[tuple(aind)]

    return b


def intensity_projection_series_all(infiles, outname, fct=np.max, normalize='all'):
    import tifffile as tiff
    from pystackreg import StackReg
    from skimage.transform import warp

    import phenotastic.file_processing as fp

    fdata = [fp.tiffload(x).data for x in infiles]
    shapes = [x.shape for x in fdata]
    max_dim = np.max(shapes)
    nchannels = fdata[0].shape[1]
    ntp = len(fdata)

    sr = StackReg(StackReg.RIGID_BODY)
    stack = np.zeros((nchannels, max_dim * ntp, 3 * max_dim))
    for chan in range(nchannels):
        cstack = np.zeros((3, max_dim * ntp, max_dim))
        for dim in range(3):
            cdstack = np.zeros((ntp, max_dim, max_dim))
            for tp in range(len(fdata)):
                one_proj = np.max(fdata[tp][:, chan], axis=dim)
                one_proj = match_shape(one_proj, (max_dim, max_dim))
                cdstack[tp] = one_proj
            tmats = sr.register_stack(cdstack, moving_average=ntp)
            for ii in range(len(tmats)):
                cdstack[ii] = warp(cdstack[ii], tmats[ii], preserve_range=True)
            cdstack = np.vstack(cdstack)
            cstack[dim] = cdstack

        if normalize == 'all':
            cstack /= np.max(cstack)
        elif normalize == 'first':
            cstack /= np.max(cstack[0])

        cstack = np.hstack(cstack)
        stack[chan] = cstack

    out = np.hstack(stack)
    out = out.astype(np.float32)
    # TODO: Save as png instead
    tiff.imsave(outname, out)


def mode(x):
    if len(x) > 0:
        return max(list(x), key=list(x).count)
    else:
        return np.nan


def car2sph(xyz):
    """
    Convert cartesian coordinates into spherical coordinates.

    Convert a list of cartesian coordinates x, y, z to spherical coordinates
    r, theta, phi. theta is defined as 0 along z-axis.

    Parameters
    ----------
    xyz :

    Returns
    -------


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


def sph2car(rtp):
    """
    Convert spherical coordinates into cartesian coordinates.

    Convert a list of spherical coordinates r, theta, phi to cartesian coordinates
    x, y, z. Theta is defined as 0 along z-axis.

    Parameters
    ----------
    rtp :


    Returns
    -------


    """
    xyz = np.zeros(rtp.shape)
    xyz[:, 0] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
    xyz[:, 1] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
    xyz[:, 2] = rtp[:, 0] * np.cos(rtp[:, 1])
    return xyz


def to_uint8(data, normalize=True):
    data = data.astype('float')
    if normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    else:
        data = data / np.max(data) * 255
    data = data.astype(np.uint8)
    return data


def matching_rows(array1, array2):
    return np.array(
        np.all((array1[:, None, :] == array2[None, :, :]), axis=-1).nonzero()
    ).T


def rand_cmap(
    nlabels,
    type='bright',
    first_color_black=True,
    last_color_black=False,
    verbose=False,
):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    import colorsys

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [
            (
                np.random.uniform(low=0.0, high=1),
                np.random.uniform(low=0.2, high=1),
                np.random.uniform(low=0.9, high=1),
            )
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(
                colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
            )

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list(
            'new_map', randRGBcolors, N=nlabels
        )

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list(
            'new_map', randRGBcolors, N=nlabels
        )

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
            spacing='proportional',
            ticks=None,
            boundaries=bounds,
            format='%1i',
            orientation='horizontal',
        )

    return random_colormap
