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
import Meristem_Phenotyper_3D as ap


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


