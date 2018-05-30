#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:53:06 2018

@author: henrik
"""
import numpy as np
import vtkInterface as vi
import domain_processing as boa
#import copy

def PlotImage(arr, res=(1, 1, 1), offset=(0, 0, 0), mask=None, **kwargs):
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
    pobj.Plot()

def PlotPointData(mesh, pointData, var='domain', boaCoords=[], show_boundaries=False,
                bcolor='black', bpsize=10, blinewidth=10, *args, **kwargs):
    pobj = vi.PlotClass()
    pobj.AddMesh(mesh, scalars=pointData[var].values, **kwargs)

    if len(boaCoords) > 0:
        pobj.AddPointLabels(np.array(boaCoords), np.array([str(ii) for ii in xrange(
            len(boaCoords))]), fontsize=30, pointcolor='w', textcolor='w')

    if show_boundaries:
        for ii in xrange(len(boaCoords)):
            pobj.AddMesh(boa.get_domain_boundary(mesh, pointData, ii),
                         color=bcolor, psize=bpsize, linethick=blinewidth)

    pobj.Plot(*args)
