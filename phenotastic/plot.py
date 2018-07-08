#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:53:06 2018

@author: henrik
"""
import numpy as np
import vtkInterface as vi
import domain_processing as boa

def coord_array(arr, res=(1, 1, 1), offset=(0, 0, 0)):
    '''
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
    '''

    xv = offset[0] + np.arange(0, arr.shape[0] * res[0], res[0])
    yv = offset[1] + np.arange(0, arr.shape[1] * res[1], res[1])
    zv = offset[2] + np.arange(0, arr.shape[2] * res[2], res[2])
    xx, yy, zz = np.array(np.meshgrid(xv, yv, zv)).transpose(0, 2, 1, 3)

    # Make compatible lists
    coords = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    return coords

def PlotImage(arr, res=(1, 1, 1), offset=(0, 0, 0), mask=None, mesh=None, meshopacity=1, **kwargs):
    '''
    Plot an intensity image with its mesh (optional).

    Parameters
    ----------
    arr : np.ndarray
        Array defining dimensions of coordinate array
    res : 3-tuple, optional
        Resolution of the array in the three dimensions. Default = (1,1,1).
    offset : 3-tuple, optional
        Origin offset. Default = (0,0,0).
    mask : np.ndarray, optional
        Array of same dimension as arr. Specifies what values should (True) or
        shouldn't be plotted. Default = None.
    mesh : vi.PolyData, optional
        PolyData mesh to plot along with the intensity array. Default = None.
    meshopacity : float, optional
        Opacity of the mesh.

    Note
    ----
    Only 3D.

    Returns
    -------
    No return. Plots input array and mesh if given.

    '''
    # Create coordinate matrix
    xv = offset[0] + np.arange(0, arr.shape[0] * res[0], res[0])
    yv = offset[1] + np.arange(0, arr.shape[1] * res[1], res[1])
    zv = offset[2] + np.arange(0, arr.shape[2] * res[2], res[2])
    xx, yy, zz = np.array(np.meshgrid(xv, yv, zv)).transpose(0, 2, 1, 3)

    # Make compatible lists
    coords = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    vals = arr.ravel().copy()

    # Filter out the data we want to see
    if mask is not None:
        mask = mask.ravel()
        coords = coords[mask]
        vals = vals[mask]

    # Plot
    pobj = vi.PlotClass()
    pobj.AddPoints(coords, scalars=vals, **kwargs)

    if mesh is not None:
        pobj.AddMesh(mesh, opacity=meshopacity)

    pobj.Plot()

def PlotPointData(mesh, pdata, var='domain', boacoords=[], show_boundaries=False,
                bcolor='black', bpsize=10, blinewidth=10, *args, **kwargs):
    '''
    Plot point data of mesh, with eventual domain boundaries and attractor maxima.

    Parameters
    ----------
    mesh : vi.PolyData
        Mesh to plot.
    pdata : pd.DataFrame
        DataFrame containing the point data of the mesh.
    var : str, optional
        Variable in pdata to visualise. Default = 'domain'.
    boacoords : list, optional
        List of the coordinates of potential basins of attraction. If given, will
        label the domain curvature maxima. Default = [].
    show_boundaries : bool, optional
        Show domain boundaries flag. Default = False.
    bcolor : str, optional
        Boundary color. Default = 'black'.
    bpsize : float, optional
        Boundary point size. Default = 10.
    blinewith : float, optional
        Boundary line width. Default = 10.

    Returns
    -------
    No return. Plots input.

    '''
    pobj = vi.PlotClass()
    pobj.AddMesh(mesh, scalars=pdata[var].values, **kwargs)

    if len(boacoords) > 0:
        pobj.AddPointLabels(np.array(boacoords), np.array([str(ii) for ii in xrange(
            len(boacoords))]), fontsize=30, pointcolor='w', textcolor='w')

    if show_boundaries:
        for ii in xrange(len(boacoords)):
            pobj.AddMesh(boa.get_domain_boundary(mesh, pdata, ii),
                         color=bcolor, psize=bpsize, linethick=blinewidth)

    pobj.Plot(**kwargs)
