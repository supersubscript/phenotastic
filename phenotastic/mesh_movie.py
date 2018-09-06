#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:33:22 2018

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
from phenotastic.external.clahe import clahe

''' FILE INPUT '''
home = os.path.expanduser('~')
fib = '/home/henrik/data/fibonacci/'
dirs = [fib + '160831-Col0-Seeds-LowN-24h-light-Mixes-Soil-Sand/1-2-Soil-1-2-Sand',
        fib + '160831-Col0-Seeds-LowN-24h-light-Mixes-Soil-Sand/1-3-Soil-2-3-Sand',
        fib + '160831-Col0-Seeds-LowN-24h-light-Mixes-Soil-Sand/1-Soil-0-Sand-LowN',
        fib + '160831-Col0-Seeds-LowN-24h-light-Mixes-Soil-Sand/2-3-Soil-1-3-Sand']

files = []
for ii in dirs:
    ff = os.listdir(ii)
    ff = map(lambda x: os.path.join(ii, x), ff)
    files.extend(ff)

#outdir = '/home/henrik/out_fib_comparison_corrected'
#if not os.path.exists(outdir):
#    os.mkdir(outdir)
#    os.mkdir(outdir + '/figs')
#m_outfile = outdir + '/meristem_data.dat'

#with open(m_outfile, 'w') as f:
#    f.writelines(np.array(['#index\t', 'fname\t', 'domain\t', 'dist_boundary\t',
#                           'dist_com\t', 'angle\t', 'area\t', 'maxdist\t',
#                           'maxdist_xy\t', 'com_coords\t', 'ismeristem\n']))

#for file_ in files:
file_ = files[24]
f = fp.tiffload(file_)
meta = f.metadata
data = f.data.astype(np.float)
resolution = fp.get_resolution(f)
fluo = data[:, 0]

''' Create AutoPhenotype object to store the data in '''
A = ap.AutoPhenotype()
A.data = fluo.copy()
A.data = A.data.astype(np.uint16)

''' Process data before creating contour. '''
A.data[A.data < 3] = 0
A.data = clahe(A.data, np.array(A.data.shape) / 8, clip_limit=10)
A.data = A.data.astype(np.float)
A.data = A.data / np.max(A.data)

for ii in xrange(1):
    A.data = median_filter(A.data, size=1)
for ii in xrange(3):
    A.data = gaussian_filter(
        A.data, sigma=[3. / (resolution[0] / resolution[1]), 3, 3])

''' Smooth the data (to fill holes) and create a contour. lambda2 > lambda1:
  more variable on inside. Smoothing might have to be corrected for different
  xy-z dimensionality. Iterations should ideally be at least 10, smoothing
  around 4. '''
A.data = (A.data - np.min(A.data)) / (np.max(A.data) - np.min(A.data))
A.data = A.data * np.max(fluo)

################################################################################
factor = .5
contour = morphological_chan_vese(A.data, iterations=10,
                                  init_level_set=A.data > factor *
                                  np.mean(A.data),
                                  smoothing=1, lambda1=1, lambda2=10)
#contour = mp.fill_contour(contour, fill_xy=False)

################################################################################
''' Run MarchingCubes in skimage and convert to VTK format '''
xyzres = resolution
A.contour = contour.copy()

verts, faces, normals, values = measure.marching_cubes_lewiner(
    A.contour, 0, spacing=list(resolution / np.min(resolution)), step_size=1,
    allow_degenerate=False)
faces = np.hstack(np.c_[np.full(faces.shape[0], 3), faces])
surf = vi.PolyData(verts, faces)

''' Process mesh '''
A.mesh = surf
A.mesh = mp.ECFT(A.mesh, 1000)

bottom_cut = 20
A.mesh = A.mesh.ClipPlane([bottom_cut, 0, 0], [1, 0, 0], inplace=False)
A.mesh = mp.ECFT(A.mesh, 100)
A.mesh = A.mesh.GenerateNormals(inplace=False)

A.mesh.RotateY(-90)
A.mesh = mp.remove_normals(A.mesh, threshold_angle=45, flip=False)
A.mesh.RotateY(90)
#    A.mesh = mp.remove_bridges(A.mesh)
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

A.mesh = A.mesh.Smooth(iterations=100, relaxation_factor=.01,
                       boundary_smoothing=False,
                       feature_edge_smoothing=False, inplace=False)
A.mesh = mp.ECFT(A.mesh, 0)

# Sufficient loop to remesh
while True:
    try:
        A.mesh = mp.remesh_decimate(A.mesh, iters=3)
        A.mesh = mp.remesh(A.mesh, A.mesh.npoints)
    except:
        print('Problem with remeshing. Attempting to clip away bottom ' +
              ' vertices.')
        A.mesh = A.mesh.ClipPlane([A.mesh.bounds[0] + 1, 0, 0],
                                  [1, 0, 0], inplace=False)
        A.mesh = mp.ECFT(A.mesh, 0)
    break

A.mesh = mp.remove_bridges(A.mesh)
A.mesh = mp.correct_bad_mesh(A.mesh)
A.mesh = mp.ECFT(A.mesh, 100)

A.mesh = mp.remove_tongues(A.mesh, radius=30, threshold=2, threshold2=.8)
A.mesh = mp.drop_skirt(A.mesh, 1000)
A.mesh = mp.remesh(A.mesh, A.mesh.npoints)
A.mesh = A.mesh.GenerateNormals(inplace=False)

###############################################################################
if np.sum(nps.vtk_to_numpy(A.mesh.GetPointData().GetNormals()), axis=0)[0] < 0:
    A.mesh.FlipNormals()
neighs = np.array([ap.get_connected_vertices(A.mesh, ii)
                   for ii in xrange(A.mesh.npoints)])

curvs = A.mesh.Curvature('mean')
curvs = boa.set_boundary_curv(curvs, A.mesh, np.min(curvs))
curvs = boa.filter_curvature(curvs, neighs, np.min, 1)
curvs = boa.filter_curvature(curvs, neighs, np.mean, 10)

#A.mesh.Plot(scalars=curvs)
#opts[:, 1] += (np.max(opts[:, 1]) - np.min(opts[:, 1])) / 2

A.mesh.RotateY(-45)
A.mesh.Translate([0, -(np.max(A.mesh.points[:, 1]) - np.min(A.mesh.points[:, 1])) / 2, 0])
plobj = vi.PlotClass()
plobj.AddMesh(A.mesh, scalars=pdata.domain)
#plobj.AddMesh(A.mesh, color='orange')
plobj.Plot(autoclose=False)

# Open a gif
plobj.OpenGif('mesh_rot_segmented.gif')

# Update Z and write a frame for each updated position
opts = A.mesh.points.copy()
degrees = 360*1
nframes = 80*1
plobj.SetBackground('white')

done1 = False
all_coords = [AxisRotation(opts, ii, axis='y') for ii in range(degrees)[::int(np.floor(degrees/nframes))]]

for ii, angle in enumerate(range(degrees)[::int(np.floor(degrees/nframes))]):
    plobj.UpdateCoordinates(all_coords[ii])
    plobj.WriteFrame()

# Close movie
plobj.Close()



