#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on 4 Aug 2017

@author: henrikahl
'''

import os
home = os.path.expanduser('~')
os.chdir(home + '/projects/Meristem_Phenotyper_3D')
import glob
import numpy as np
import vtk
import Meristem_Phenotyper_3D as ap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from handy_functions import squared_dist_para, paraboloid, close_window
from tissueviewer.tvtiff import tiffread  # tiffsave
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import hist


"""
FILE INPUT
"""
#directory = home + '/projects/Meristem_Phenotyper_3D/test_images/meristem_test/'
#filenames = sorted(os.listdir(directory))
#filenames = glob.glob(directory + 'results.csv')
filename = home + "/plant2/00hrs_plant2_trim-acylYFP_hmin_2_asf_1_s_1.50_clean_3.tif"
#filename = "/home/henrikahl/data/yi_curvature/tiffs/Col0 (0-5um)-1.tif"
#filename = "/home/henrikahl/data/yi_curvature/tiffs/1-2017mmdd-prc1-1 (0-5um)-4_sam.tif"

# USE THESE
A = ap.AutoPhenotype()
##A.data, _ = tiffread(home + '/plant2/processed_tiffs/' + filename)
##A.data, _ = tiffread(home + '/plant2/' + filename)
#A.data, _ = tiffread(filename)
#
#A.contour_fit_threshold(threshold=.5, smooth_iterate=3)
##A.reduce(factor=5, spline=False)
#A.mesh_conversion()
#A.smooth_mesh(300, .5)
#A.clean_mesh()
#A.curvature_slice(0, 'mean')
##A.fill_holes(50)
#A.clean_mesh()
#A.feature_extraction(20)  # Only keep features > 20 % of pixels
#A.sphere_fit()
#A.sphere_evaluation()
#A.paraboloid_fit_mersitem(weighted=False)

A.load2(filename[:-4] + "_output")

results = A.results
meristem = results.filter(like='para_p').iloc[0]
alpha, beta, gamma = results.filter(['para_alpha', 'para_beta',
                                     'para_gamma']).iloc[0]
p1, p2, p3, p4, p5 = meristem

data = np.zeros(((A.mesh.GetNumberOfPoints()), 3))
for ii in xrange(0, A.mesh.GetNumberOfPoints()):
  data[ii] = A.mesh.GetPoints().GetPoint(ii)

data = ap.rot_coord(data, [alpha, beta, gamma])
pa   = np.array([data[:,0], data[:,1], paraboloid(data[:,0], data[:,1], meristem)]).T

scores = np.zeros((100, 100))
sdist  = np.linspace(-1000, 1000, 100)
scurv  = np.linspace(-0.1, 0.1, 100)

A.show_paraboloid_and_mesh()

for ii in xrange(len(sdist)):
  for jj in xrange(len(scurv)):
    scores[ii, jj] = np.sum((p1 - scurv[jj]) * (data[:, 0] - sdist[ii])**2
                + (p2 - scurv[jj]) * (data[:, 1])**2
                + p3 * (data[:, 0] - sdist[ii])
                + p4 * (data[:, 1])
                + p5 - data[:, 2])**2

smallest = np.sort(scores.flatten())[:5]
index = []
for ii in smallest:
  index.append(zip(*np.where(scores == ii)))
index = np.reshape(index, (5,2))



p1 = p1 + sdist[0]
p3


scores = scores.T
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('z translation')
plt.ylabel('a|b modification')
#ax.set_yticklabels(np.linspace(-50,50, 6))
#ax.set_xticklabels(np.linspace(-0.001,0.003, 6))
out = plt.imshow(np.log(scores + 0.0001), cmap='hot')
cbar = fig.colorbar(out, label = 'log(cost)')
plt.show()
squared_dist_para(data, meristem, shiftx=0)
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
upperBound = 2000
lowerBound = -2000
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
