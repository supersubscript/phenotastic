#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:34:01 2017

@authors: maxbrambach, henrikahl
"""

import numpy as np
import pandas as pd
import vtk
import automated_phenotyping as ap
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist

def paraboloid(x,y,p):
    p1,p2,p3,p4,p5 = p
    return p1*x**2+p2*y**2+p3*x+p4*y+p5

def swaprows(a,how=[2,0,1]):
    a[:,[0,1,2]] = a[:,how]
    return a

def radius(x,y):
    return np.sqrt(x**2+y**2)

def sort_columns(a):
    for i in range(np.shape(a)[0]):
        if a[i,0] < a[i,1]:
            a[i,[0,1]] = a[i,[1,0]]
    return a

"""
FILE INPUT
"""

directory = '/home/maxbrambach/plants_henrik/results/plant15/top75' # change plantXX
filenames = sorted(os.listdir(directory))
clv_raw = pd.read_csv('/home/maxbrambach/workspace/automated_phenotyping/topcoords_tomax.dat',sep=None,header=None,engine='python')
# clv_raw = clv_raw.as_matrix([4,5,6])[20:40,:] # plant02
# clv_raw = clv_raw.as_matrix([4,5,6])[537:557,:] # plant04
# clv_raw = clv_raw.as_matrix([4,5,6])[980:1002,:] # plant13
clv_raw = clv_raw.as_matrix([4,5,6])[1506:1528,:] # plant15
clv_raw = swaprows(clv_raw)

conv_factors_clv0     =  [1./0.26 ,1./0.2396150, 1./0.2396150]
conv_factors_clv_rest =  [1./0.26 ,1./0.2196471, 1./0.2196471]

clv_raw[0,:]  = clv_raw[0,:]  * conv_factors_clv0
clv_raw[1:,:] = clv_raw[1:,:] * conv_factors_clv_rest
par_raw = []
clv = []
parameters = []
# print clv_raw
# exit()

# exit()
data = []

for i in range(len(filenames)):
    _temp = pd.read_csv(directory+'/'+filenames[i])
    p = _temp[['para_p1','para_p2','para_p3','para_p4','para_p5']]
    p = p.as_matrix()[0,:]
    angl = _temp[['para_alpha','para_beta','para_gamma']]
    angl = angl.as_matrix()[0,:]
#     angl[-1] = 0
    paracoord = _temp[['para_apex_x','para_apex_y','para_apex_z']]
    paracoord = paracoord.as_matrix()[0,:]
    paracoord = ap.rot_coord(np.array([paracoord]), angl)[0]
    p[[2,3,4]] = [0,0,0]
    clvcoord = ap.rot_coord(np.array([clv_raw[i,:]]),angl,False)[0]
    clv.append(clvcoord)
    par_raw.append(paracoord)
    parameters.append(abs(p[[0,1]]))
    data.append([p,paracoord,clvcoord])
par_raw = np.array(par_raw)
clv_raw = clv
# print par_raw
# par_raw = swaprows(par_raw)
# print par_raw
parameters = sort_columns(np.array(parameters))
print parameters
# exit()

dataset = 2
if dataset == 0:
    scale = conv_factors_clv0[-1]**-1
else:
    scale = conv_factors_clv_rest[-1]**-1

#create flat parboloid
flatpar = np.zeros((512,512))
for x in range(np.shape(flatpar)[0]):
    for y in range(np.shape(flatpar)[1]):
        flatpar[x,y] = abs(paraboloid(x-256, y-256, data[dataset][0]))

norm_data = clv_raw-par_raw

radii = radius(norm_data[:,1]*scale, norm_data[:,0]*scale)

bins = np.linspace(0, 50, 10)

hist, bins = np.histogram(radii, 8,range=(0,55))
center = (bins[:-1] + bins[1:]) / 2


"""
PLOTTER FOR THE PARABOLOID CONTOUR + HISTOGRAM
"""

plt.figure()
# plt.grid()
plt.contour(np.array(range(-255,257))*scale,np.array(range(-255,257))*scale,-flatpar*.26,10,zorder=0)
plt.colorbar(label='$z$ [$\mu$m]')
plt.plot(0,0,'ro',label='Geometric apex')
plt.bar(center,hist*5,width=5,zorder=1,label='Peak density')
# plt.scatter(norm_data[:,1]*scale, norm_data[:,0]*scale,100,facecolors='none',edgecolors='C1',zorder=2, label='CLV3 peak')
plt.scatter(radius(norm_data[:,1]*scale,
                   norm_data[:,0]*scale),np.zeros((len(radii),)),
            100,facecolors='none',edgecolors='C1',zorder=2, label='CLV3 peak')
plt.legend(loc=8)
plt.xlabel('$x$ [$\mu$m]')
plt.ylabel('$y$ [$\mu$m]')
plt.xlim((-55,55))
plt.ylim((-55,55))
plt.show()


# f = plt.figure()
#
# plot = f.add_subplot(111)
# plt.grid()
# # plt.scatter(parameters[:,0],parameters[:,1])
# plot.plot(np.linspace(0, 76, len(parameters[:,0])),parameters[:,0],'o',label='$a$')
# plot.plot(np.linspace(0, 76, len(parameters[:,0])),parameters[:,1],'d',label='$b$')
# plot.text(1, .0016, '$ax^2+by^2=z$',size=18)
# plt.xlabel('time [h]')
# plt.ylabel('parameter values $a,b$')
# # plot.set_xticklabels([])
# plt.legend()
# plt.show()


exit()
apexCLV = data[0][2]
apexFIT = data[0][1]

p1, p2, p3, p4, p5 = data[0][0]
# print p1

# exit()
# p1 = 0.0037280971
# p2 = 0.0041441133
# p3 = 0.5212172086
# p4 = 3.2206246432
# p5 = 521.064076481

''' PARABOLOID PLOT 3D '''
quadric = vtk.vtkQuadric()
quadric.SetCoefficients(p1,p2,0,0,0,0,p3,p4,-1,p5)
# F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2 + a3*x*y + a4*y*z + a5*x*z + a6*x + a7*y + a8*z + a9
sample = vtk.vtkSampleFunction()
sample.SetSampleDimensions(100, 100, 100)
sample.SetImplicitFunction(quadric)
upperBound = 2000
lowerBound = -500
sample.SetModelBounds(lowerBound, upperBound, lowerBound, upperBound, lowerBound, upperBound)

contour = vtk.vtkContourFilter()
contour.SetInputConnection(sample.GetOutputPort())
contour.GenerateValues(1, 0, 0)

polyPlot = [contour.GetOutput()]

# apexFIT = np.array([np.array([76.309681938,297.467555099,276.894275673])])
# apexCLV = np.array([np.array([6.4788003,14.181592,16.076298])*conv_factors_clv])
# print apexCLV
# angles = [1.6521003084,-0.9883916016,149.176195692]

sphereSourceFIT = vtk.vtkSphereSource()
sphereSourceFIT.SetCenter(apexFIT[0], apexFIT[1], apexFIT[2])
sphereSourceFIT.SetRadius(5.)
polyPlot.append(sphereSourceFIT.GetOutput())
sphereSourceCLV = vtk.vtkSphereSource()
sphereSourceCLV.SetCenter(apexCLV[0], apexCLV[1], paraboloid(apexCLV[0], apexCLV[1],data[0][0]))#apexCLV[2])
sphereSourceCLV.SetRadius(5.)
polyPlot.append(sphereSourceCLV.GetOutput())
# print apexCLV
# exit()
ap.view_polydata(polyPlot,(1,1,1))
