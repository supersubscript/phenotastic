#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:41:10 2018

@author: henrik
"""

import vtk
import pandas as pd
import tifffile as tiff
from scipy.ndimage.morphology import binary_fill_holes
from skimage.exposure import equalize_hist, equalize_adapthist
import phenotastic.plot as pl
import os
import numpy as np
import phenotastic.Meristem_Phenotyper_3D as ap
import copy
from skimage import measure
from vtk.util import numpy_support as nps
#from scipy.ndimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.segmentation import morphological_chan_vese
import vtkInterface as vi
from vtkInterface.common import AxisRotation
import phenotastic.domain_processing as boa
import phenotastic.mesh_processing as mp
import phenotastic.file_processing as fp
#from phenotastic.external.clahe import clahe
from phenotastic.misc import mkdir, listdir, autocrop


''' FILE INPUT '''
home = os.path.expanduser('~')
#dir_ = '/home/henrik/data/from-marcus/'
dir_ = '/home/henrik/data/fibonacci/161010-COl0-LowN-24h-light-Soil-Sand/1-1-Soil/'

files = listdir(dir_, include='.lsm')
files.sort()
#files = files[20:]

file_ = files[0]
import gc

gc.collect()
f = fp.tiffload(file_)
meta = f.metadata
data = f.data.astype(np.float)
del f
data = autocrop(data, 30, fct=np.max)

resolution = np.array([meta['voxelsizez'], meta['voxelsizey'], meta['voxelsizex']])
resolution *= 1e6

fluo = data.copy()
fluo = fluo[:,0]

fluo /= np.max(fluo)
fluo *= np.iinfo(np.uint16).max


''' Create AutoPhenotype object to store the data in '''
A = ap.AutoPhenotype()
A.data = fluo
A.data = A.data.astype(np.uint16)

''' Process data before creating contour. '''
from scipy.ndimage import generate_binary_structure
import mahotas
#A.data[A.data < mahotas.otsu(A.data, ignore_zeros=True) / 10.] = 0

footprint = generate_binary_structure(3, 2).astype('bool')
for ii in xrange(2):
    A.data = median_filter(A.data, footprint=footprint)
A.data = gaussian_filter(
    A.data, sigma=[3. / resolution[0] * 0.,
                   3. / resolution[1] * 0.25,
                   3. / resolution[2] * 0.25])
A.data = gaussian_filter(
    A.data, sigma=[3. / resolution[0] * 0,
                   3. / resolution[1] * 0.25/2,
                   3. / resolution[2] * 0.25/2])
A.data = gaussian_filter(
    A.data, sigma=[3. / resolution[0] * 0,
                   3. / resolution[1] * 0.25/3,
                   3. / resolution[2] * 0.25/3])

#A.data[A.data < mahotas.otsu(A.data, ignore_zeros=True) / 4] = 0

''' Smooth the data (to fill holes) and create a contour. lambda2 > lambda1:
  more variable on inside. Smoothing might have to be corrected for different
  xy-z dimensionality. Iterations should ideally be at least 10, smoothing
  around 4. '''
factor = .5
contour = morphological_chan_vese(A.data, iterations=10,
                                  init_level_set=A.data > factor * np.mean(A.data[A.data > 0]),
                                  smoothing=1, lambda1=1, lambda2=1)

#        from skimage.morphology import binary_fill_holes
for ii in xrange(contour.shape[0]):
    contour[ii] = binary_fill_holes(contour[ii])

################################################################################
''' Run MarchingCubes in skimage and convert to VTK format '''
xyzres = resolution
A.contour = contour

verts, faces, normals, values = measure.marching_cubes_lewiner(
    A.contour, 0, spacing=list(resolution / np.min(resolution)), step_size=1,
    allow_degenerate=False)
faces = np.hstack(np.c_[np.full(faces.shape[0], 3), faces])
A.mesh = vi.PolyData(verts, faces)

del faces, verts, normals, values

A.mesh.RotateY(-90)
A.mesh = mp.remove_normals(A.mesh, threshold_angle=45, flip=False)
A.mesh.RotateY(90)

''' Process mesh '''
bounds = A.mesh.GetBounds()
A.mesh.ClipPlane([np.ceil(bounds[0]), 0, 0], [(xyzres[0] + 0.0001), 0, 0])
A.mesh.ClipPlane([np.floor(bounds[1]), 0, 0], [-(xyzres[0] + 0.0001), 0, 0])
A.mesh.ClipPlane([0, np.ceil(bounds[2]), 0], [0, (xyzres[1] + 0.0001), 0])
A.mesh.ClipPlane([0, np.floor(bounds[3]), 0], [0, -(xyzres[1] + 0.0001), 0])
A.mesh.ClipPlane([0, 0, np.ceil(bounds[4])], [0, 0, (xyzres[2] + 0.0001)])
A.mesh.ClipPlane([0, 0, np.floor(bounds[5])], [0, 0, -(xyzres[2] + 0.0001)])
A.mesh = mp.ECFT(A.mesh, 100.)
A.mesh = mp.correct_bad_mesh(A.mesh)


bottom_cut = 10
A.mesh = A.mesh.ClipPlane([bottom_cut, 0, 0], [1, 0, 0], inplace=False)
A.mesh = mp.ECFT(A.mesh, 100)
A.mesh = A.mesh.GenerateNormals(inplace=False)

A.mesh = mp.remove_bridges(A.mesh)
A.mesh = mp.ECFT(A.mesh, 100)
A.mesh = A.mesh.GenerateNormals(inplace=False)

################################################################################
if np.sum(nps.vtk_to_numpy(A.mesh.GetPointData().GetNormals()), axis=0)[0] < 0:
    A.mesh.FlipNormals()

A.mesh = A.mesh.Decimate(
    0.95, volume_preservation=True, normals=True, inplace=False)
A.mesh = mp.ECFT(A.mesh, 0)
A.mesh = mp.remove_tongues(A.mesh, radius=30, threshold=2, threshold2=.8)
A.mesh = mp.ECFT(A.mesh, 100)

A.mesh = A.mesh.ClipPlane([bottom_cut, 0, 0], [1, 0, 0], inplace=False)
A.mesh = mp.ECFT(A.mesh, 100)

A.mesh = mp.remove_bridges(A.mesh)
A.mesh = mp.ECFT(A.mesh, 100)

A.mesh = mp.correct_bad_mesh(A.mesh)
A.mesh = mp.drop_skirt(A.mesh, 1000)

A.mesh = A.mesh.Smooth(iterations=100, relaxation_factor=.1,
                       boundary_smoothing=False,
                       feature_edge_smoothing=False, inplace=False)
A.mesh = mp.ECFT(A.mesh, 0)

# Sufficient loop to remesh
while True:
    try:
#        A.mesh = mp.remesh_decimate(A.mesh, iters=3)
        A.mesh = mp.remesh(A.mesh, A.mesh.npoints)
    except:
        print('Problem with remeshing. Attempting to clip away bottom ' +
              ' vertices.')
        A.mesh = A.mesh.ClipPlane([A.mesh.bounds[0] + 1, 0, 0],
                                  [1, 0, 0], inplace=False)
        A.mesh = mp.ECFT(A.mesh, 0)
    break

for ii in xrange(1):
    A.mesh = mp.ECFT(A.mesh, 0)
    A.mesh = mp.correct_bad_mesh(A.mesh)
    A.mesh = mp.remesh(A.mesh, A.mesh.npoints)

A.mesh = mp.remove_bridges(A.mesh)
A.mesh = mp.correct_bad_mesh(A.mesh)
A.mesh = mp.ECFT(A.mesh, 100)

A.mesh = mp.remove_tongues(A.mesh, radius=30, threshold=2, threshold2=.8)
A.mesh = mp.drop_skirt(A.mesh, 1000)
A.mesh = mp.correct_bad_mesh(A.mesh)

A.mesh = mp.remesh(A.mesh, A.mesh.npoints)
A.mesh = A.mesh.GenerateNormals(inplace=False)

if np.sum(nps.vtk_to_numpy(A.mesh.GetPointData().GetNormals()), axis=0)[0] < 0:
    A.mesh.FlipNormals()
neighs = np.array([ap.get_connected_vertices(A.mesh, ii)
                   for ii in xrange(A.mesh.npoints)])

curvs = A.mesh.Curvature('mean')
try:
    curvs = boa.set_boundary_curv(curvs, A.mesh, np.min(curvs))
except:
    pass
curvs = boa.filter_curvature(curvs, neighs, np.min, 2)
curvs = boa.filter_curvature(curvs, neighs, np.mean, 10)
A.mesh.Plot(scalars=curvs)

###############################################################################

pdata = boa.init_pointdata(A, curvs, neighs)

''' Identify BoAs'''
pdata = boa.domains_from_curvature(pdata)
boas, boasData = boa.get_boas(pdata)
print boa.nboas(pdata)

''' Process boas '''
safeCopy = copy.deepcopy(pdata)
pdata = copy.deepcopy(safeCopy)
boas, boasData = boa.get_boas(pdata)

pdata = boa.merge_boas_depth(A, pdata, threshold=0.01)
boas, boasData = boa.get_boas(pdata)
print boa.nboas(pdata)

pdata = boa.merge_boas_distance(pdata, boas, boasData, 15)
boas, boasData = boa.get_boas(pdata)
print boa.nboas(pdata)

pdata = boa.merge_boas_engulfing(A, pdata, threshold=0.6)
boas, boasData = boa.get_boas(pdata)
print boa.nboas(pdata)

#    pdata = boa.remove_boas_size(pdata, .05, method="relative_largest")
#    boas, boasData = boa.get_boas(pdata)
#    print boa.nboas(pdata)

''' Visualise '''
print boa.nboas(pdata)
print boa.boas_npoints(pdata)
boas, boasData = boa.get_boas(pdata)
boacoords = np.array([tuple(ii) for ii in boasData[['z', 'y', 'x']].values])

pl.PlotPointData(A.mesh, pdata, 'domain',
                 boacoords=boacoords, show_boundaries=True)

#####################
distance = 15
radius = (distance / np.sqrt(3) + 0.05 * distance)
shift = float(distance) / 2

zv = np.arange(bounds[0], bounds[1] - 0.0001, distance/2)
yv = np.arange(bounds[2], bounds[3] - 0.0001, distance)
xv = np.arange(bounds[4], bounds[5] - 0.0001, distance)

zz, yy, xx = np.array(np.meshgrid(zv, yv, xv)).transpose(0, 2, 1, 3)
for ii in xrange(xx.shape[0]):
    if ii % 2:
        yy[ii] += float(distance) / 2
    else:
        xx[ii] += float(distance) / 2

coords = np.vstack((zz.ravel(), yy.ravel(), xx.ravel())).T

ipd = vtk.vtkImplicitPolyDataDistance()
ipd.SetInput(A.mesh)
ipd.Modified()

dists = np.zeros((len(coords),))
pts = np.zeros((len(coords), 3))
for ii in xrange(len(coords)):
    dists[ii] = ipd.EvaluateFunctionAndGetClosestPoint(coords[ii], pts[ii])

coords = coords[dists > 0]

pobj = vi.PlotClass()
#pobj.AddMesh(A.mesh, color = 'red')
#pobj.AddPoints(coords, opacity=1, color='red')

actors = []
for cc in coords:
    spherevtk = vtk.vtkSphereSource()
    spherevtk.SetCenter(cc)
    spherevtk.SetRadius(radius)
    spherevtk.Update()
    spoly = vi.PolyData(spherevtk.GetOutput())
    smapper = vtk.vtkPolyDataMapper()
    smapper.SetInputData(spoly)
    sactor = vtk.vtkActor()
    sactor.GetProperty().SetOpacity(1)
    sactor.SetMapper(smapper)
    actors.append(sactor)
actors = np.array(actors)
#pobj.AddActor(sactor)
#pobj.Plot()

sink = coords[:, 0] == np.min(zz)
l1 = dists[dists >= 0] < (distance + 0.000001)
#l1 =

for ii, act in enumerate(actors):
    if sink[ii]:
        act.GetProperty().SetColor(.5,0,0)
    elif l1[ii]:
        act.GetProperty().SetColor(.1, 1, 0)
    pobj.AddActor(act)
#pobj.Plot()

df = pd.DataFrame()
df['x'] = coords[:, 2]
df['y'] = coords[:, 1]
df['z'] = coords[:, 0]
df['r'] = radius
df['sink'] = sink.astype('float')
df['l1'] = l1.astype('float')








