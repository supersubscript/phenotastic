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
import pandas as pd
import copy
import handy_functions as hf
import attractor_processing as boa
import networkx as nx
from skimage import measure
from vtk.util import numpy_support as nps
home = os.path.expanduser('~')

''' FILE INPUT '''
dir_ = os.path.abspath(os.path.join(home, 'projects', 'Meristem_Phenotyper_3D'))
file_ = os.path.abspath(os.path.join(dir_, 'test_images', 'meristem_test.tif'))
#file_ = os.path.abspath(os.path.join(home, 'home3', 'From-Benoit', 'parameter_optimisation_images', 'Col0-24h-light-HighN-10.tif'))
#file_ = os.path.abspath(os.path.join(home, 'data', 'plant2', "00hrs_plant2_trim-acylYFP_hmin_2_asf_1_s_1.50_clean_3.tif")) # NPA meristem
#file_ = os.path.abspath(os.path.join(home, 'data', "20171102-FM4-64-cmu1cmu2-2-15_sam.tif")) # CMU1CMU2 mutant
#file_ = os.path.abspath(os.path.join(home, 'data', "20171103-FM4-64-Col0-2-15_sam.tif")) # WT mutant
#file_ = os.path.abspath(os.path.join(home, 'data', "C2-WUS-GFP-24h-light-1-1-Soil-1.tif")) # One of Benoit's
#file_ = os.path.abspath(os.path.join(home, 'data', "C2-WUS-GFP-24h-light-2-3-Soil-1-3-Sand-9.tif")) # Another of Benoits. Worse quality.

# USE THESE
A = ap.AutoPhenotype()
A.read_data(file_)
A.contour_fit_threshold(threshold=1.0, smooth_iterate=25)

verts, faces, normals, values = measure.marching_cubes(
    A.contour, 0, spacing=(1.0, 1.0, 1.0))

points = vtk.vtkPoints()
points.SetData(nps.numpy_to_vtk(np.ascontiguousarray(verts), array_type=vtk.VTK_FLOAT, deep=True))

# Create polygons
nFaces = len(faces)
faces = np.array([np.append(len(ii), ii) for ii in faces]).flatten()
polygons = vtk.vtkCellArray()
polygons.SetCells(nFaces, nps.numpy_to_vtk(faces, array_type=vtk.VTK_ID_TYPE))

# Create polydata from points and polygons
polygonPolyData = vtk.vtkPolyData()
polygonPolyData.SetPoints(points)
polygonPolyData.SetPolys(polygons)
polygonPolyData.Update()

###########
A.mesh = polygonPolyData
#A.show_mesh()
A.compute_normals()
A.clean_mesh()
A.smooth_mesh(iterations=0, relaxation_factor=.1)
A.quadric_decimation(dec=2000.0, method="npoints") # retain 1 %
A.clean_mesh()
A.smooth_mesh(iterations=300, relaxation_factor=.01)

#A.smooth_mesh(iterations=300, relaxation_factor=.01)
A.update_mesh()
A.clean_mesh()
#A.show_mesh()
#A.invert_normals()
#A.show_normals()
#ap.save_polydata_ply(A.mesh, file_[:-4])


#A.show_mesh()
dist_ = 2
curvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'max'], operations=['-'], ignore_boundary=True)
#A.show_curvatures(stdevs="all", curvs=curvs)

''' Create graphs '''
nPoints = A.mesh.GetNumberOfPoints()
coords = [A.mesh.GetPoint(ii) for ii in xrange(A.mesh.GetNumberOfPoints())]
neighs = [ap.get_connected_vertices(A.mesh, ii) for ii in xrange(nPoints)]
coords = pd.DataFrame(coords)
neighs = pd.DataFrame(np.array(neighs))
pointData = pd.concat([curvs, coords, neighs], axis=1)
pointData.columns = ['curv', 'x', 'y', 'z', 'neighs']
pointData['domain'] = np.NaN * nPoints

''' Identify BoAs'''
pointData      = boa.domains_from_curvature(pointData)
boas, boasData = boa.get_boas(pointData)

''' Process boas '''
safeCopy = copy.deepcopy(pointData)
pointData = copy.deepcopy(safeCopy)
boas, boasData = boa.get_boas(pointData)

pointData = boa.merge_boas_depth(pointData, threshold=0.010)
#pointData = boa.merge_boas_depth(pointData, threshold=0.010)
boas, boasData = boa.get_boas(pointData)

pointData = boa.merge_boas_engulfing(A, pointData, threshold=0.6)
boas, boasData = boa.get_boas(pointData)

pointData = boa.remove_boas_size(pointData, .02, method = "relative_all")
boas, boasData = boa.get_boas(pointData)

''' Visualise in VTK '''
print boa.get_nBoas(pointData)
print boa.get_boas_nPoints(pointData)
#A.show_curvatures(stdevs = "all", curvs = curvs)
boaCoords = [tuple(x) for x in boasData[['x', 'y', 'z']].values]
A.show_point_values(vals = pd.DataFrame(np.array(pointData['domain'])),
                    discrete=True, bg=[0, 0, 0])
