#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:53:06 2018

@author: henrik
"""
import sys, os
# os.chdir('/home/henrik/projects/surface_extraction/code/phenotastic/phenotastic')

import numpy as np
import phenotastic.domains as boa
import pyvista
#import phenotastic.Meristem_Phenotyper_3D as ap

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

    xv = offset[0] + np.arange(0, arr.shape[0] * res[0] - 0.000001, res[0])
    yv = offset[1] + np.arange(0, arr.shape[1] * res[1]- 0.000001, res[1])
    zv = offset[2] + np.arange(0, arr.shape[2] * res[2]- 0.000001, res[2])
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

def PlotImage(arr, res=(1, 1, 1), offset=(0, 0, 0), mask=None, mesh=None, mopacity=1,
              popacity=1, psize=5, bg=[.5, .5, .5], pcolor=None, prng=None, pname='',
              pstitle='', pflipscalars=False, pcolormap=None, pncolors=256, mcolor=None,
              mstyle=None, mscalars=None, mrng=None, mstitle=None, mshowedges=True,
              mpsize=5.0, mlinethick=None, mflipscalars=False, mlighting=False,
              mncolors=256, minterpolatebeforemap=False, mcolormap=None, mkwargs=dict(),
              return_pobj=False):
    '''
    Plot an intensity image with its mesh (optional).
    TODO: Rewrite documentation

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
    mesh : vista.PolyData, optional
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
    pobj = pyvista.Plotter(notebook=False)
    pobj.add_points(coords, color=pcolor, psize=psize, scalars=vals, rng=prng, name=pname,
                   opacity=popacity, stitle=pstitle, flipscalars=pflipscalars, colormap=pcolormap,
                   ncolors=pncolors)
    pobj.set_background(bg)

    if mesh is not None:
        pobj.add_mesh(mesh, color=mcolor, style=mstyle, scalars=mscalars, rng=mrng,
                     stitle=mstitle, showedges=mshowedges, psize=mpsize, opacity=mopacity,
                     linethick=mlinethick, flipscalars=mflipscalars, lighting=mlighting,
                     interpolatebeforemap=minterpolatebeforemap, ncolors=mncolors,
                     colormap=mcolormap, **mkwargs)

    if return_pobj:
        return pobj
    else:
        pobj.plot()

def PlotPointData(mesh, pdata, var='domain', boacoords=[], show_boundaries=False,
                bcolor='black', bpsize=10, blinewidth=10, *args, **kwargs):
    '''
    Plot point data of mesh, with eventual domain boundaries and attractor maxima.

    Parameters
    ----------
    mesh : vista.PolyData
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
    pobj = pyvista.Plotter()
    pobj.add_mesh(mesh, scalars=pdata[var].values, **kwargs)

    if len(boacoords) > 0:
        pobj.add_point_labels(np.array(boacoords), np.array([str(ii) for ii in range(
            len(boacoords))]), font_size=30, point_color='w', text_color='w')

    if show_boundaries:
        for ii in range(len(boacoords)):
            pobj.add_mesh(boa.get_domain_boundary(mesh, pdata, ii),
                         color=bcolor, point_size=bpsize, line_width=blinewidth)

    pobj.plot(**kwargs)

def rot_matrix_44(angles, invert=False):
    alpha, beta, gamma = angles
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha), 0],
                   [0, np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 0, 1]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta), 0],
                   [0, 1, 0, 0],
                   [-np.sin(beta), 0, np.cos(beta), 0],
                   [0, 0, 0, 1]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                   [np.sin(gamma), np.cos(gamma), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    if invert == True:
        R = np.linalg.inv(np.matmul(np.matmul(Rz, Ry), Rx))
    elif invert == False:
        R = np.matmul(np.matmul(Rz, Ry), Rx)

    return R

def PlotParaboloid(mesh, p, sampleDim=(200, 200, 200),
                   bounds=[-2000, 2000] * 3):
    p1, p2, p3, p4, p5, alpha, beta, gamma = p

    import vtk
    # Configure
    quadric = vtk.vtkQuadric()
    quadric.SetCoefficients(p1, p2, 0, 0, 0, 0, p3, p4, -1, p5)

    sample = vtk.vtkSampleFunction()
    sample.SetSampleDimensions(sampleDim)
    sample.SetImplicitFunction(quadric)
    sample.SetModelBounds(bounds)
    sample.Update()

    contour = vtk.vtkContourFilter()
    contour.SetInputData(sample.GetOutput())
    contour.Update()

    contourMapper = vtk.vtkPolyDataMapper()
    contourMapper.SetInputData(contour.GetOutput())
    contourActor = vtk.vtkActor()
    contourActor.SetMapper(contourMapper)

    rotMat = rot_matrix_44([alpha, beta, gamma], invert=True)
    trans = vtk.vtkMatrix4x4()
    for ii in range(0, rotMat.shape[0]):
        for jj in range(0, rotMat.shape[1]):
            trans.SetElement(ii, jj, rotMat[ii][jj])

    transMat = vtk.vtkMatrixToHomogeneousTransform()
    transMat.SetInput(trans)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(contour.GetOutput())
    transformFilter.SetTransform(transMat)
    transformFilter.Update()
    tpoly = pyvista.pointset.PolyData(transformFilter.GetOutput())
    tpoly.clip_with_plane([mesh.bounds[0] - 20, 0, 0], [1, 0, 0])

    pobj = pyvista.Plotter()
    pobj.add_mesh(tpoly, opacity=.5, showedges=False, color='orange')
    pobj.add_mesh(mesh, opcaity=.9, color='green')
    pobj.plot()
