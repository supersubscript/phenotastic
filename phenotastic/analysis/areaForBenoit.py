#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on 4 Aug 2017

@author: henrikahl
'''

import os
import numpy as np
import vtk
import Meristem_Phenotyper_3D as ap
import scipy
import pandas as pd
import copy
import itertools
import handy_functions as hf
import networkx as nx
home = os.path.expanduser('~')

def calculate_curvature(A, curvature_type='mean'):
    polyAlg = A.mesh.GetProducerPort()
    curvature = vtk.vtkCurvatures()
    curvature.SetInputConnection(polyAlg)
    if curvature_type == 'max':
        curvature.SetCurvatureTypeToMaximum()
        A.curvature_type = 'Maximum_Curvature'
    elif curvature_type == 'min':
        curvature.SetCurvatureTypeToMinimum()
        A.curvature_type = 'Minimum_Curvature'
    elif curvature_type == 'mean':
        curvature.SetCurvatureTypeToMean()
        A.curvature_type = 'Mean_Curvature'
    elif curvature_type == 'gauss':
        curvature.SetCurvatureTypeToGaussian()
        A.curvature_type = 'Gauss_Curvature'

    curvature.Update()
    return curvature.GetOutput()

def invert_normals(mesh):
      reverse = vtk.vtkReverseSense()
      reverse.SetInput(mesh)
      reverse.ReverseCellsOn()
      reverse.ReverseNormalsOn()
      return reverse.GetOutput()

def get_connected_vertices(mesh, seed):
    connectedVertices = []
    cellIdList = vtk.vtkIdList()
    mesh.GetPointCells(seed, cellIdList)

    # Loop through each cell using the seed point
    for ii in xrange(cellIdList.GetNumberOfIds()):
        cell = mesh.GetCell(cellIdList.GetId(ii))		# get current cell

        # Loop through the edges of the point and add all points on these.
        for e in xrange(cell.GetNumberOfEdges()):
            pointIdList = cell.GetEdge(e).GetPointIds()

            # if one of the points on the edge are the vertex point, add the
            # other one
            if pointIdList.GetId(0) == seed:
                temp = pointIdList.GetId(1)
                connectedVertices.append(temp)
            elif pointIdList.GetId(1) == seed:
                temp = pointIdList.GetId(0)
                connectedVertices.append(temp)

    return np.unique(connectedVertices)

def get_curvatures(A, dist = 1, curv_types=['mean'], operations = []):
  #  A.mesh = calculate_curvature(A, curvature_type=curvature_type)
  A.calculate_curvatures(curv_types=curv_types, operations=operations)
  curvs = A.mesh.GetPointData().GetArray(A.curvature_type)
  curvs = pd.DataFrame(vtk.util.numpy_support.vtk_to_numpy(curvs))

  if dist == 0:
    return A, curvs

  ''' Get neighbourhood '''
  # Get all connected points for each point
  neighs = np.zeros(A.mesh.GetNumberOfPoints(), dtype='object')
  for ii in xrange(A.mesh.GetNumberOfPoints()):
    neighs[ii] = get_connected_vertices(A.mesh, ii)

  # find neighbours neighbours etc.
  for nn in xrange(dist - 1):
    preNeighs = copy.deepcopy(neighs)
    for ii in xrange(len(preNeighs)):
        # find neighbours of neighbours and merge them to single list, then take union with already accounted for
        neighsNeighs = np.unique(list(itertools.chain.from_iterable(preNeighs[preNeighs[ii]])))
        neighs[ii] = np.union1d(neighs[ii], neighsNeighs)

  ''' Average curvatures '''
  preCurvs = copy.deepcopy(curvs)
  for ii in xrange(len(curvs)):
    curvs.iloc[ii] = np.mean(preCurvs.iloc[neighs[ii]])

  return A, curvs

"""
FILE INPUT
"""
dir_ = os.path.abspath(os.path.join(home, 'data', 'from-benoit'))
file_ = os.path.abspath(os.path.join(dir_, 'C2-WUS-GFP-24h-light-1-1-Soil-9.tif'))
# C2-WUS-GFP-24h-light-1-1-Soil-9.lsm
# C2-WUS-GFP-24h-light-1-3-Soil-2-3-Sand-9.lsm
# C2-WUS-GFP-24h-light-1-1-Soil-Fertilizer-9.lsm
# C2-WUS-GFP-24h-light-2-3-Soil-1-3-Sand-9.lsm
# C2-WUS-GFP-24h-light-1-2-Soil-1-2-Sand-9.lsm

# USE THESE
A = ap.AutoPhenotype()
A.read_data(file_)
A.contour_fit_threshold(threshold=.99, smooth_iterate=3)
A.mesh_conversion()
A.clean_mesh()
A.triangulate()
A.quadric_decimation(dec=0.9) # retain 1 %
A.smooth_mesh(iterations=500, relaxation_factor=.1)
A.update_mesh()
A.invert_normals()
A.update_mesh()
A.curvature_slice(0, curvature_type='mean')

#A.show_normals(maxPoints=1e5)
dist_ = 1
A, curvs = get_curvatures(A, dist=dist_, curv_types=['gauss', 'max'], operations=['-'])
#A.fill_holes(50)
A.show_mesh()
A.show_normals(reverseNormals=False, maxPoints=1e3)
A.show_curvatures(stdevs=2, curvs = curvs)
#A, curvs = get_curvatures(A, dist=2, curvature_type='mean')
#A.show_curvatures(curvature_type='mean', stdevs=2, curvs = curvs)

