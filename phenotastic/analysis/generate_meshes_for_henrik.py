#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:11:11 2018

@author: henrik
"""

import os
import numpy as np
import vtk
import Meristem_Phenotyper_3D as ap
import pandas as pd
import copy
import handy_functions as hf
from vtk.util import numpy_support as nps
import tifffile as tiff
from scipy.ndimage.morphology import binary_opening, binary_closing
from scipy.ndimage.filters import gaussian_filter
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, inverse_gaussian_gradient, checkerboard_level_set
from scipy.ndimage.morphology import binary_fill_holes
home = os.path.expanduser('~')


file_ = '/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/C3_pWUS-3XVENUS-pCLV3-mCherry-off_NPA-1-18h-2.0-750-0.5-770_topcrop_deconv.tif'
data = tiff.imread(file_)
fluo = data[:, 0]

''' Create object '''
A = ap.AutoPhenotype()
A.data = fluo

''' Smooth the data (to fill holes) and create a contour. lambda2 > lambda1:
  more variable on inside. Smoothing might have to be corrected for different
  xy-z dimensionality. Iterations should ideally be at least 10, smoothing
  around 4. '''
A.data = gaussian_filter(A.data, sigma=[3, 3, 3])
A.data = gaussian_filter(A.data, sigma=[3, 3, 3])
A.data = gaussian_filter(A.data, sigma=[3, 3, 3])
contour = morphological_chan_vese(A.data.astype(np.uint8), iterations=1,
                                  init_level_set=A.data > 1. * np.mean(A.data),
                                  smoothing=1, lambda1=1, lambda2=100)

''' Remove top slice. Normally we have a lot of noise here. Then fill holes
    inside contour. '''
A.contour = copy.deepcopy(contour)
A.contour[np.array([len(A.contour)-1])] = 0
A.contour = binary_fill_holes(A.contour)
A.contour = A.contour.astype(np.float)

from skimage import measure
''' Generate mesh '''
verts, faces, normals, values = measure.marching_cubes_lewiner(
    A.contour, 0, spacing=(0.22, 0.2516, 0.2516), step_size=1,
    allow_degenerate=True)

''' Port mesh to VTK '''
# Assign point data
points = vtk.vtkPoints()
points.SetData(nps.numpy_to_vtk(np.ascontiguousarray(verts),
                                array_type=vtk.VTK_FLOAT, deep=True))

# Create polygons
nFaces = len(faces)
faces = np.array([np.append(len(ii), ii) for ii in faces]).flatten()
polygons = vtk.vtkCellArray()
polygons.SetCells(nFaces, nps.numpy_to_vtk(faces, array_type=vtk.VTK_ID_TYPE))

# Create polydata from points and polygons
polygonPolyData = vtk.vtkPolyData()
polygonPolyData.SetPoints(points)
polygonPolyData.SetPolys(polygons)

''' Process mesh '''
#A.show_mesh(opacity=5)


nPoints = [3, 10, 100, 1000, 10000, 100000]
for ii in nPoints:
  # Create object
  A.mesh = polygonPolyData
  A.clean_mesh()
  A.compute_normals()
  A.clean_mesh()
  A.smooth_mesh(iterations=0, relaxation_factor=.1)

  # Decimate it
  A.quadric_decimation(dec=ii, method="npoints")
  A.mesh.Modified()
#  A.mesh.Update()

  A.clean_mesh()
  A.mesh.Modified()
#  A.mesh.Update()

  A.compute_normals()
  A.mesh.Modified()
#  A.mesh.Update()

  ap.save_polydata_ply(A.mesh, "mesh_" + str(A.mesh.GetNumberOfPoints()) + '.ply')
  ap.save_polydata_vtk(A.mesh, "mesh_" + str(A.mesh.GetNumberOfPoints()) + '.xml')

  print A.mesh.GetNumberOfPoints()
