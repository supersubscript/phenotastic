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
from vtk.util import numpy_support as nps
home = os.path.expanduser('~')

''' FILE INPUT '''
dir_ = os.path.abspath(os.path.join(home, 'projects', 'Meristem_Phenotyper_3D'))
file_ = os.path.abspath(os.path.join(dir_, 'test_images', 'meristem_test.tif'))
#file_ = os.path.abspath(os.path.join(home, 'data', 'plant2', "00hrs_plant2_trim-acylYFP_hmin_2_asf_1_s_1.50_clean_3.tif")) # NPA meristem
#file_ = os.path.abspath(os.path.join(home, 'data', "20171102-FM4-64-cmu1cmu2-2-15_sam.tif")) # CMU1CMU2 mutant
#file_ = os.path.abspath(os.path.join(home, 'data', "20171103-FM4-64-Col0-2-15_sam.tif")) # WT mutant
#file_ = os.path.abspath(os.path.join(home, 'data', "C2-WUS-GFP-24h-light-1-1-Soil-1.tif")) # One of Benoit's
#file_ = os.path.abspath(os.path.join(home, 'data', "C2-WUS-GFP-24h-light-2-3-Soil-1-3-Sand-9.tif"))

# USE THESE
from tissueviewer.tvtiff import tiffread, tiffsave

A = ap.AutoPhenotype()
A.data, metaData = tiffread(file_)
A.contour_fit_threshold(threshold=.95, smooth_iterate=3)
A.mesh_conversion()
A.clean_mesh()
A.triangulate()
A.quadric_decimation(dec=0.5) # retain 1 %
A.smooth_mesh(300, 0.1)

A.curvature_slice(0, curv_types=['mean'], operations="", lower=True)
#A.fill_holes(50)
A.clean_mesh()
A.show_mesh()
A.feature_extraction(20)  # Only keep features > 20 % of pixels
A.sphere_fit()
A.sphere_evaluation()
A.paraboloid_fit_mersitem(weighted=False)

#A.load2(filename[:-4] + "_output")

results = A.results
meristem = results.filter(like='para_p').iloc[0]
alpha, beta, gamma = results.filter(['para_alpha', 'para_beta',
                                     'para_gamma']).iloc[0]
p1, p2, p3, p4, p5 = meristem



#data = np.zeros(((A.mesh.GetNumberOfPoints()), 3))
#for ii in xrange(0, A.mesh.GetNumberOfPoints()):
#  data[ii] = A.mesh.GetPoints().GetPoint(ii)
#
#data = ap.rot_coord(data, [alpha, beta, gamma])
#pa = np.array([data[:,0], data[:,1], paraboloid(data[:,0], data[:,1], meristem)]).T
#
#scores = np.zeros((100, 100))
#sdist = np.linspace(-1000, 1000, 100)
#scurv = np.linspace(-0.1, 0.1, 100)
#
#for ii in xrange(len(sdist)):
#  for jj in xrange(len(scurv)):
#    scores[ii][jj] = squared_dist_para(data, meristem, shiftx=sdist[ii], shiftcurv=scurv[jj])
#
#smallest = np.sort(scores.flatten())[:5]
#index = []
#for ii in smallest:
#  print(zip(*np.where(scores == ii)))
#
#scores = scores.T
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.xlabel('z translation')
#plt.ylabel('a|b modification')
##ax.set_yticklabels(np.linspace(-50,50, 6))
##ax.set_xticklabels(np.linspace(-0.001,0.003, 6))
#out = plt.imshow(np.log(scores + 0.0001), cmap='hot')
#cbar = fig.colorbar(out, label = 'log(cost)')
#plt.show()
#squared_dist_para(data, meristem, shiftx=0)
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#mesh = ax.scatter(data[:,0], data[:,1],data[:,2], cmap=plt.cm.coolwarm,
#                       linewidth=0, antialiased=False, alpha = .01)
#para = ax.scatter(pa[:,0], pa[:,1], paraboloid(pa[:,0], pa[:,1], meristem), cmap=plt.cm.coolwarm,
#                       linewidth=0, antialiased=False, alpha = .01)
#ax.set_zlim(-100, 500)
#ax.set_xlim(-100, 500)
#ax.set_ylim(-100, 500)
#ax.view_init(45, 60)
#plt.show()

# TODO: GLÖM INTE ATT ROTERA SKITEN!


''' PARABOLOID PLOT 3D '''
# Configure
quadric = vtk.vtkQuadric()
quadric.SetCoefficients(p1, p2, 0, 0, 0, 0, p3, p4, -1, p5)
# Quadric.SetCoefficients(p1, p2, 0, 0, 0, 0, p3, p4, -1, p5)
# F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2 + a3*x*y + a4*y*z + a5*x*z + a6*x + a7*y

# + a8*z + a9
sample = vtk.vtkSampleFunction()
sample.SetSampleDimensions(100, 100, 100)
sample.SetImplicitFunction(quadric)
upperBound = 500
lowerBound = -500
sample.SetModelBounds([lowerBound, upperBound] * 3)

# Create the paraboloid contour
contour = vtk.vtkContourFilter()
contour.SetInputConnection(sample.GetOutputPort())
contour.GenerateValues(1, 1, 1)
contour.Update()
contourMapper = vtk.vtkPolyDataMapper()
contourMapper.SetInput(contour.GetOutput())
contourMapper.SetScalarRange(0.0, 1.2)
contourActor = vtk.vtkActor()
contourActor.SetMapper(contourMapper)
contourActor.GetProperty().SetOpacity(0.1)

rotMat = ap.rot_matrix_44([alpha, beta, gamma], invert=True)
trans = vtk.vtkMatrix4x4()
for ii in xrange(0, rotMat.shape[0]):
    for jj in xrange(0, rotMat.shape[1]):
        trans.SetElement(ii, jj, rotMat[ii][jj])

transMat = vtk.vtkMatrixToHomogeneousTransform()
transMat.SetInput(trans)
transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetInputConnection(contour.GetOutputPort())
transformFilter.SetTransform(transMat)
transformFilter.Update()

transformedMapper = vtk.vtkPolyDataMapper()
transformedMapper.SetInputConnection(transformFilter.GetOutputPort())
transformedActor = vtk.vtkActor()
transformedActor.SetMapper(transformedMapper)
transformedActor.GetProperty().SetColor(0, 1, 0)
transformedActor.GetProperty().SetOpacity(0.2)

# Input sphere with top coordinates for paraboloid (corrected)
sphereSource = vtk.vtkSphereSource()
sphereSource.SetCenter(results['para_apex_x'][0], results[
                       'para_apex_y'][0], results['para_apex_z'][0])
sphereSource.SetRadius(1)
sphereSource.Update()
sphereMapper = vtk.vtkPolyDataMapper()
sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
sphereActor = vtk.vtkActor()
sphereActor.SetMapper(sphereMapper)

# Input the mesh
meshData = A.mesh
meshMapper = vtk.vtkPolyDataMapper()
meshMapper.SetInput(meshData)
meshActor = vtk.vtkActor()
meshActor.SetMapper(meshMapper)
meshActor.GetProperty().SetOpacity(0.9)

''' Plotting '''
# Setup the window
ren1 = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren1)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

ren1.AddActor(meshActor)
ren1.AddActor(sphereActor)
ren1.AddActor(transformedActor)
ren1.SetBackground(.1, .2, .3)  # Background color white

# Render and interact
renWin.Render()
iren.Start()
hf.close_window(iren)
del renWin, iren
