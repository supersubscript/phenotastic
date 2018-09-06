#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:00:11 2018

@author: henrik
"""

import pickle
import numpy as np
import time
import scipy.optimize as opt
import vtk
from vtk.util import numpy_support as nps
#import phenotastic.Meristem_Phenotyper_3D as ap
import os

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
    """ Flatten a list of lists """
    return [item for sublist in llist for item in sublist]

def remove_empty_slices(arr, keepaxis=0):
    """ Remove empty slices (based on the total intensity) in an ndarray """
    not_empty = np.sum(arr, axis=tuple(np.delete(range(arr.ndim), keepaxis))) > 0
    arr = arr[not_empty]
    return arr

def reject_outliers(data, n=2.):
    """ Remove outliers outside of n standard deviations.

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
    s = d / (mdev if mdev else 1.)
    filtered_data = data[s < n]
    return filtered_data

def angle_difference(ang1, ang2, period=360):
    ''' Return the smallest angle difference on a template with periodic boundary conditions.

    Returns
    -------
    angs : np.array of floats
        The smallest angles.

    '''
    difference = np.subtract(ang1, ang2)
    angs = np.array([np.abs(np.mod(difference, period)), np.abs(np.mod(difference, -period))])
    angs = np.min(angs, axis=0)
    return angs

def divergence_angles(angles, period=360):
    ''' Divergence angles between an ordered list of angles '''
    div_angs = angle_difference(angles,
                                       np.roll(angles, 1), period=period)[1:]

    return div_angs

def paraboloid(x, y, p):
    """ Return the z-value for a paraboloid given input xy-coordinates and
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
    """ Get colors with maximal inter-color contrast.

    Parameters
    ----------
    n : Numbers of RGB colors to return.
         (Default value = 64)

    Returns
    -------
    rgbs : list
        List of colours (RGB) up to a certain n that maximise contrast.

    """
    rgbs = [[0, 0, 0],
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
            [232, 94, 190]]
    return rgbs[0:n]

import math


def prime_sieve(n, output={}):
    '''
    Return a dict or a list of primes up to N create full prime sieve for
    N=10^6 in 1 sec

    '''

    nroot = int(math.sqrt(n))
    sieve = range(n+1)
    sieve[1] = 0

    for i in xrange(2, nroot+1):
        if sieve[i] != 0:
            m = n/i - i
            sieve[i*i: n+1:i] = [0] * (m+1)

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
    '''
    Get a list of all factors for N.

    Example
    -------
    >>> get_factors(10)
    >>> Out[1]: [1, 2, 5, 10]

    '''
    if primelist is None:
        primelist = prime_sieve(n,output=[])

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
        exp = [i**(x+1) for x in range(fcount[i])]
        for j in exp:
            level.extend([j*x for x in factors])
        factors.extend(level)

    return factors



def get_prime_factors(n, primelist=None):
    ''' Get a list of prime factors and corresponding powers.

    Example
    -------
    >>> get_prime_factors(140) # 140 = 2^2 * 5^1 * 7^1
    >>> Out[1]: [(2, 2), (5, 1), (7, 1)]

    '''
    if primelist is None:
        primelist = prime_sieve(n,output=[])

    fs = []
    for p in primelist:
        count = 0
        while n % p == 0:
            n /= p
            count += 1
        if count > 0:
            fs.append((p, count))

    return fs


def autocrop(arr, threshold=8e7, fct=np.sum):
    sumarr = np.max(arr, axis=1)

    cp = np.zeros((sumarr.ndim, 2), dtype=np.int)
    for ii in xrange(sumarr.ndim):
        summers = np.array([0, 1, 2])[np.array([0, 1, 2]) != ii]

        vals = fct(sumarr, axis=tuple(summers))
        first = next((e[0] for e in enumerate(vals) if e[1] > threshold), 0)
        last = len(
            vals) - next((e[0] for e in enumerate(vals[::-1]) if e[1] > threshold), 0) - 1

        cp[ii] = first, last

    return arr[cp[0, 0]:cp[0, 1], :, cp[1, 0]:cp[1, 1], cp[2, 0]:cp[2, 1]]



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def listdir(path, include=None, exclude=None, full=True):
    files = os.listdir(path)
    files = np.array(files)

    if full:
        files = np.array(map(lambda x: os.path.join(path, x), files))

    # Include
    if isinstance(include, str):
        files = np.array(filter(lambda x: include in x, files))
    elif isinstance(include, (list, np.ndarray)):
        matches = np.array([np.array([inc in ii for ii in files]) for inc in include])
        matches = np.any(matches, axis=0)
        files = files[matches]

    # Exclude
    if isinstance(exclude, str):
        files = np.array(filter(lambda x: exclude not in x, files))
    elif isinstance(exclude, (list, np.ndarray)):
        matches = np.array([np.array([exc in ii for ii in files]) for exc in exclude])
        matches = np.logical_not(np.any(matches, axis=0))
        files = files[matches]

    return files

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
                't shape must have the same number of dimensions as the input')
    except TypeError:
        raise TypeError('t must be array-like')

    try:
        if isinstance(val, (int, long, float, complex)):
            b = np.ones(t, a.dtype) * val
        elif val == 'max':
            b  = np.ones(t, a.dtype) * np.max(a)
        elif val == 'mean':
            b  = np.ones(t, a.dtype) * np.mean(a)
        elif val == 'median':
            b  = np.ones(t, a.dtype) * np.median(a)
        elif val == 'min':
            b  = np.ones(t, a.dtype) * np.min(a)
    except TypeError:
        raise TypeError('Pad value must be numeric or string')
    except ValueError:
        raise ValueError('Pad value must be scalar or valid string')

    aind = [slice(None, None)] * a.ndim
    bind = [slice(None, None)] * a.ndim

    # pad/trim comes after the array in each dimension
    if side == 'after':
        for dd in xrange(a.ndim):
            if a.shape[dd] > t[dd]:
                aind[dd] = slice(None, t[dd])
            elif a.shape[dd] < t[dd]:
                bind[dd] = slice(None, a.shape[dd])
    # pad/trim comes before the array in each dimension
    elif side == 'before':
        for dd in xrange(a.ndim):
            if a.shape[dd] > t[dd]:
                aind[dd] = slice(int(a.shape[dd] - t[dd]), None)
            elif a.shape[dd] < t[dd]:
                bind[dd] = slice(int(t[dd] - a.shape[dd]), None)
    # pad/trim both sides of the array in each dimension
    elif side == 'both':
        for dd in xrange(a.ndim):
            if a.shape[dd] > t[dd]:
                diff = (a.shape[dd] - t[dd]) / 2.
                aind[dd] = slice(int(np.floor(diff)), int(a.shape[dd] - np.ceil(diff)))
            elif a.shape[dd] < t[dd]:
                diff = (t[dd] - a.shape[dd]) / 2.
                bind[dd] = slice(int(np.floor(diff)), int(t[dd] - np.ceil(diff)))
    else:
        raise Exception('Invalid choice of pad type: %s' % side)

    b[bind] = a[aind]

    return b