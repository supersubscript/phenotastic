#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on 4 Aug 2017

@author: henrikahl
'''

import os
import numpy as np
#import vtk
import Meristem_Phenotyper_3D as ap
import pandas as pd
import handy_functions as hf
from vtk.util import numpy_support as nps
home = os.path.expanduser('~')


"""
FILE INPUT
"""
dir_ = os.path.abspath(os.path.join(home, 'projects', 'Meristem_Phenotyper_3D'))
file_ = os.path.abspath(os.path.join(dir_, 'test_images', 'meristem_test.tif'))
#file_ = os.path.abspath(os.path.join(home, 'data', 'C3-pWUS-pCLV3-High-Quality-780.tif'))


# USE THESE
A = ap.AutoPhenotype()
A.read_data(file_)
A.contour_fit_threshold(threshold=.95, smooth_iterate=3)
A.mesh_conversion()
A.clean_mesh()
A.triangulate()
A.quadric_decimation(dec=0.99) # retain 1 %
A.smooth_mesh(iterations=3000, relaxation_factor=.001)
#A.invert_normals()
A.update_mesh()

#A.show_normals()
#A.curvature_slice(0, curv_types=["min"])
#A.show_mesh()

dist_ = 2
#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max'],   ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min'],   ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean'],  ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss'], ignore_boundary=True)

''' Subtr '''
#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'min'], operations=['-'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'min'], operations=['-'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'min'], operations=['-'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'min'], operations=['-'], ignore_boundary=True)

#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'max'], operations=['-'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'max'], operations=['-'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'max'], operations=['-'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'max'], operations=['-'], ignore_boundary=True)

#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'gauss'], operations=['-'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'gauss'], operations=['-'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'gauss'], operations=['-'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'gauss'], operations=['-'], ignore_boundary=True)

#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'mean'], operations=['-'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'mean'], operations=['-'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'mean'], operations=['-'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'mean'], operations=['-'], ignore_boundary=True)

''' Mult '''
#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'min'], operations=['*'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'min'], operations=['*'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'min'], operations=['*'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'min'], operations=['*'], ignore_boundary=True)
#
#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'max'], operations=['*'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'max'], operations=['*'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'max'], operations=['*'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'max'], operations=['*'], ignore_boundary=True)

#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'gauss'], operations=['*'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'gauss'], operations=['*'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'gauss'], operations=['*'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'gauss'], operations=['*'], ignore_boundary=True)

#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'mean'], operations=['*'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'mean'], operations=['*'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'mean'], operations=['*'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'mean'], operations=['*'], ignore_boundary=True)

''' Div '''
#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'min'], operations=['/'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'min'], operations=['/'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'min'], operations=['/'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'min'], operations=['/'], ignore_boundary=True)

#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'max'], operations=['/'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'max'], operations=['/'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'max'], operations=['/'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'max'], operations=['/'], ignore_boundary=True)

#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'gauss'], operations=['/'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'gauss'], operations=['/'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'gauss'], operations=['/'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'gauss'], operations=['/'], ignore_boundary=True)

#maxCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['max',   'mean'], operations=['/'], ignore_boundary=True)
#minCurvs   = ap.get_curvatures(A, dist_=dist_, curv_types=['min',   'mean'], operations=['/'], ignore_boundary=True)
#meanCurvs  = ap.get_curvatures(A, dist_=dist_, curv_types=['mean',  'mean'], operations=['/'], ignore_boundary=True)
#gaussCurvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'mean'], operations=['/'], ignore_boundary=True)

''' Deviation from flatness '''
dist_ = 2
H = ap.get_curvatures(A, dist_=dist_, curv_types=['mean'],  ignore_boundary=True)
K = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss'], ignore_boundary=True)
curvs = pd.DataFrame(np.sqrt((4*H**2 - 2*K**2)[0]))
A.show_curvatures(stdevs="all", curvs = curvs, return_actors=False)



std = "all"
actMaxCurvs   = A.show_curvatures(stdevs=std, curvs = maxCurvs,   return_actors=True)
actMinCurvs   = A.show_curvatures(stdevs=std, curvs = minCurvs,   return_actors=True)
actMeanCurvs  = A.show_curvatures(stdevs=std, curvs = meanCurvs,  return_actors=True)
actGaussCurvs = A.show_curvatures(stdevs=std, curvs = gaussCurvs, return_actors=True)

hf.render_four_viewports([actMaxCurvs, actMinCurvs, actMeanCurvs, actGaussCurvs], [2,3,0,1])

#curvs = maxCurvs

#A.curvature_slice(0)
#A.fill_holes(10)
#A.mesh.Update()
#A.show_mesh()
#A.show_normals(reverseNormals=False)
#A.show_curvatures(curvature_type='mean', stdevs="all")

#A.show_curvatures(stdevs="all", curvs = curvs)