#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:32:28 2018

@author: henrik
"""
import phenotastic.file_processing as fp
import tifffile as tiff
import scipy.optimize as opt
import stackAlign.external.transformations as tf

im1_path = '/home/henrik/data/180129-pWUS-3X-VENUS-pCLV3-mCherry-Timelapse/on_NPA/pWUS-3X-VENUS-pCLV3-mCherry-on-NPA-1-0h-mCherry-0.7-Gain-800-5um.lsm'
im2_path = '/home/henrik/data/180129-pWUS-3X-VENUS-pCLV3-mCherry-Timelapse/on_NPA/pWUS-3X-VENUS-pCLV3-mCherry-on-NPA-1-0h-mCherry-0.7-Gain-800.lsm'

im1 = fp.tiffload(im1_path).data
im2 = fp.tiffload(im2_path).data

# 2d
im1 = im1[5, 0, ...]
im2 = im2[33, 0, ...]

im1 = im1[:256]

# 3d
im1 = im1[:, 0, ...]
im2 = im1[:]
#im2 = im2[:, 0, ...]

from sklearn.metrics import mutual_info_score
import numpy as np


def mutual_information(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def rigid2D(moving, target, init=(0, 0, 0), method='Powell', verbose=True):
    from skimage.transform import EuclideanTransform
    from skimage.transform import warp
    import numpy as np

    moving = match_shape(moving, target.shape, side='both', val=0)

    def errfunc(p, moving_, verbose):
        dx, dy, theta = p
        mat = EuclideanTransform(translation=(dx, dy), rotation=theta)
        warped = warp(moving_, mat, preserve_range=True)
        cost = -np.corrcoef(warped.ravel(), target.ravel())[0, 1]

        if verbose:
            print(cost, warped.max(), target.max())

        return cost

    out = opt.minimize(errfunc, init, args=(moving, verbose), method=method,
                       options=dict(disp=verbose),
                       bounds=((-moving.shape[0], moving.shape[0]),
                               (-moving.shape[1], moving.shape[1]),
                               (-90, 90)))
    dx, dy, theta = out.x

    trsf_img = warp(moving,
                    EuclideanTransform(translation=(dx, dy), rotation=theta),
                    preserve_range=True)

    return trsf_img, out.x


def align_stack(img, init=(0, 0, 0), method='Powell', verbose=True):

    if img.ndim != 3:
        raise Exception('Input image must be 3D')

    for ii in xrange(1, img.shape[0]):
        img[ii], _ = rigid2D(img[ii], img[ii - 1], init=init,
                             method=method, verbose=verbose)

    return img


def align_timeseries(img, target_slice=0, init=(0, 0, 0), method='Powell', verbose=True):

    for ii in xrange(img.shape[0]):
        img[ii], _ = rigid2D(img[ii], target_slice, init=init,
                             method=method, verbose=verbose)

    return img


def match_shape(a, t, side='both', val=0):
    """
    Forces an array to a t size by either padding it with a constant or
    truncating it.

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

##############

def rigid3D(moving, target, init=(0, 0, 0, 0, 0, 0), method='Powell', verbose=True):
    from skimage.transform import warp
    from scipy.ndimage.interpolation import affine_transform
    import numpy as np

    moving = match_shape(moving, target.shape, side='both', val=0)

    def errfunc(p, moving_, verbose):
        dx, dy, dz, alpha, beta, gamma = p
        mat = tf.compose_matrix(translate=[dx, dy, dz], angles=[alpha, beta, gamma])
        warped = affine_transform(moving, mat)
        cost = -np.corrcoef(warped.ravel(), target.ravel())[0, 1]

        if verbose:
            print(cost, warped.max(), target.max())

        return cost

    out = opt.minimize(errfunc, init, args=(moving, verbose), method=method,
                       options=dict(disp=verbose))
    dx, dy, dz, alpha, beta, gamma = out.x

    trsf_img = affine_transform(moving,
                    tf.compose_matrix(translate=[dx, dy, dz], angles=[alpha, beta, gamma]))

    return trsf_img, out.x












