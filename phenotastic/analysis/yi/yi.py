#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Created on 4 Aug 2017

@author: henrikahl
'''

import os
os.chdir('/home/henrikahl/projects/Meristem_Phenotyper_3D')
import glob
import numpy as np
import vtk
import Meristem_Phenotyper_3D as ap
import handy_functions as hf
from tissueviewer.tvtiff import tiffread  # tiffsave
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import hist

"""
FILE INPUT
"""
#directory = '/home/henrikahl/projects/Meristem_Phenotyper_3D/test_images/meristem_test/'
#filenames = sorted(os.listdir(directory))
#filenames = glob.glob(directory + 'results.csv')
filename = '/home/henrikahl/home3/yi_curvature/cu4 (0-25 um)-1.tif'


# USE THESE
A = ap.AutoPhenotype()
#A.data, _ = tiffread('/home/henrikahl/plant2/processed_tiffs/' + filename)
A.data, _ = tiffread(filename)
A.contour_fit_threshold(threshold=.8, smooth_iterate=3)
#A.reduce(factor=5, spline=False)
A.mesh_conversion()
A.smooth_mesh(300, .5)
A.clean_mesh()
A.curvature_slice(0, 'mean')
A.feature_extraction(10.)  # Only keep features > 20 % of pixels
#A.sphere_fit()
#A.sphere_evaluation()
A.paraboloid_fit_mersitem(weighted=False)

results = A.results
meristem = results.filter(like='para_p').iloc[0]
alpha, beta, gamma = results.filter(['para_alpha', 'para_beta',
                                     'para_gamma']).iloc[0]
p1, p2, p3, p4, p5 = meristem

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
upperBound = 1000
lowerBound = -1000
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
meshActor.GetProperty().SetOpacity(0.2)

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
ren1.SetBackground(.5, .5, .5)  # Background color white

# Render and interact
renWin.Render()
iren.Start()
close_window(iren)
del renWin, iren
