#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 14:37:33 2018

@author: henrik
"""
import functools
import argparse
import os
os.chdir('/home/henrik/projects/stackAlign/')
import multiprocessing
from misc import mkdir
import numpy as np
import imageProc as ip
import subprocess
import background_extraction as bg
import tifffile as tiff
#from alignment import z_correction
import copy
from alignment import z_correction
import pandas as pd

import os
import numpy as np
import vtk
import Meristem_Phenotyper_3D as ap
import pandas as pd
import copy
import handy_functions as hf
import attractor_processing as boaa
import networkx as nx
from skimage import measure
from vtk.util import numpy_support as nps
import tifffile as tiff
from scipy.ndimage.morphology import binary_opening, binary_closing
from scipy.ndimage.filters import gaussian_filter
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, inverse_gaussian_gradient, checkerboard_level_set
from scipy.ndimage.morphology import binary_fill_holes

INPUT_PATH      = '/home/henrik/filipa/'
OUTPUT_PATH     = INPUT_PATH
STACKALIGN_PATH = os.path.abspath('')

RAPID_ID = '-5um'
REFERENCE_ID = 'reference'

BG_EXT_THRES = 2
BG_EXT_ITER = 10
NPROCESSORS = 3
#r_default   = [515, 545, 580, 633, 654, 735]
#DECONV_ITERS    = [10, 8, 3]
#WAVELENGTHS = [1,2,3]
NUM_APERTURE = 1.0
REFR_INDEX = 1.333
WAVELENGTHS = [514, 514, 561]
CROP_THRES = 0
#RANGES      = np.reshape(args.range, (len(args.range) / 2, 2))
#WAVELENGTHS = np.mean(RANGES, axis=1)

''' DIRECTORIES AND FILES '''
STRETCH_FILE    = os.path.join(OUTPUT_PATH, 'stretching.txt')
ALIGNMENT_FILE  = os.path.join(OUTPUT_PATH, 'alignment.txt')
TIFFS_PATH      = os.path.join(OUTPUT_PATH, 'tiffs')
CROP_PATH       = os.path.join(OUTPUT_PATH, 'cropped')
DECONV_PATH     = os.path.join(OUTPUT_PATH, 'deconv')
FILTERED_PATH   = os.path.join(OUTPUT_PATH, 'filtered')
MIPS_PATH       = os.path.join(OUTPUT_PATH, 'mips')
PSF_PATH        = os.path.join(OUTPUT_PATH, 'PSFs')
PSF_CONFIG_PATH = os.path.join(PSF_PATH,    'configs')

''' SOFTWARE '''
DECONVOLUTIONLAB_PATH = os.path.join(STACKALIGN_PATH, 'jars',
                                     'DeconvolutionLab_2.jar')
PSF_GENERATOR_PATH    = os.path.join(STACKALIGN_PATH, 'jars',
                                     'PSFGenerator.jar')

################################################################################
##############################  RUN STUFF  #####################################
################################################################################
''' Create output directories '''
print('Creating output directories...')
mkdir(OUTPUT_PATH)
mkdir(TIFFS_PATH)
mkdir(CROP_PATH)
mkdir(FILTERED_PATH)

mkdir(PSF_PATH)
mkdir(PSF_CONFIG_PATH)
mkdir(DECONV_PATH)
mkdir(MIPS_PATH)
for ii in xrange((len(WAVELENGTHS))):
  mkdir(os.path.join(OUTPUT_PATH, 'channel%d' % (ii + 1)))

################################################################################
'''' Converting to tiffs '''
print('Converting to tiffs...')
lfiles = os.listdir(INPUT_PATH)
lfiles = [x for x in lfiles if x.endswith('.lsm')]
lfiles = map(lambda x: os.path.join(INPUT_PATH, x), lfiles)

p = multiprocessing.Pool(NPROCESSORS)
p.map(functools.partial(ip.lsm_to_tiff, output_dir=TIFFS_PATH,
                        extra_metadata=dict(excitation_wavelength=WAVELENGTHS,
                                            num_aperture=NUM_APERTURE,
                                            refr_index=REFR_INDEX)), lfiles)

tfiles      = os.listdir(TIFFS_PATH)
tfiles      = map(lambda x: os.path.join(TIFFS_PATH, x), tfiles)
tfiles      = [x for x in tfiles if x.endswith('.tif') or x.endswith('.tiff')]
tfiles      = np.sort(tfiles)

################################################################################
''' Run croptop '''
p.map(functools.partial(ip.topcrop, output_dir=CROP_PATH, threshold=CROP_THRES),
      tfiles)
cfiles      = os.listdir(CROP_PATH)
cfiles      = map(lambda x: os.path.join(CROP_PATH, x), cfiles)
cfiles      = np.sort(cfiles)

################################################################################
print(['Correcting for stretching...'])
sfiles = filter(lambda x: RAPID_ID not in x, cfiles)
p.close(); p.join(); del p
p = multiprocessing.Pool(NPROCESSORS)

################################################################################
''' Split channels '''
print('Splitting channels...')
import imageProc as ip
p.map(functools.partial(ip.split_channels, output_path=OUTPUT_PATH), sfiles)

################################################################################
''' Perform deconvolution '''
ch1 = os.listdir(OUTPUT_PATH + '/channel1')
ch2 = os.listdir(OUTPUT_PATH + '/channel2')
ch3 = os.listdir(OUTPUT_PATH + '/channel3')
ch1 = map(lambda x: os.path.join(OUTPUT_PATH, "channel1", x), ch1)
ch2 = map(lambda x: os.path.join(OUTPUT_PATH, "channel2", x), ch2)
ch3 = map(lambda x: os.path.join(OUTPUT_PATH, "channel3", x), ch3)

all_files_channels = ch1 + ch2 + ch3
all_files_channels.sort()
cfiles = copy.deepcopy(all_files_channels)
cfiles = filter(lambda x: RAPID_ID not in x, cfiles)

p.close(); p.join(); del p

p = multiprocessing.Pool(NPROCESSORS)
#import deconv
c1 = filter(lambda x: 'C1' in x, all_files_channels)
c2 = filter(lambda x: 'C2' in x, all_files_channels)
c3 = filter(lambda x: 'C3' in x, all_files_channels)

#p.map(functools.partial(deconv.deconvolve_file_to_file, output_dir = DECONV_PATH, iterations=10, method='LR'), c1)
#p.map(functools.partial(deconv.deconvolve_file_to_file, output_dir = DECONV_PATH, iterations=8, method='LR'), c2)
#p.map(functools.partial(deconv.deconvolve_file_to_file, output_dir = DECONV_PATH, iterations=5, method='LR'), c3)

#p.map(deconvolve, all_files_channels)
################################################################################
''' Correct metadata '''
#dfiles = os.listdir(DECONV_PATH)
#dfiles = map(lambda x: os.path.join(DECONV_PATH, x), dfiles)
#dfiles = np.sort(dfiles)
#cfiles = np.sort(cfiles)
#range_ = xrange(len(cfiles))
#
#def correct_metadata(ii, cfiles, dfiles):
#  f1 = tiff.TiffFile(cfiles[ii])
#  f2 = tiff.TiffFile(dfiles[ii])
#  meta = f1.imagej_metadata
#  meta['max'] = f2.imagej_metadata['max']
#  meta['min'] = f2.imagej_metadata['min']
#  tiff.imsave(dfiles[ii], f2.asarray(), imagej=True, metadata=meta)
#
#print(['Correcting metadata...'])
#p.close(); p.join(); del p
#p = multiprocessing.Pool(NPROCESSORS)
#p.map(functools.partial(correct_metadata, cfiles=cfiles, dfiles=dfiles), range_)

################################################################################
''' Run background extraction '''
def filter_bg(fname, OUTPUT_DIR, BG_EXT_THRES, BG_EXT_ITER):
  fnamebase = os.path.basename(fname)
  outname = os.path.join(OUTPUT_DIR,
                         os.path.splitext(fnamebase)[0] + '_filtered.tif')
  file_ = tiff.TiffFile(fname)
  data  = np.array(file_.asarray(), dtype='uint16')

  if data.ndim == 5:
    data = data[:, 0, 0]
#    data = data[:, 0]

#  data[data < 0] = np.iinfo('uint16').max
  meta = file_.imagej_metadata

  contour = bg.contour_fit_threshold(data,
                                     threshold      = BG_EXT_THRES,
                                     smooth_iterate = BG_EXT_ITER)
  data[contour] = 0
  meta['max'] = np.max(data)
  meta['min'] = np.min(data)

  tiff.imsave(outname, data, imagej=True, metadata=meta)

print(['Filtering background...'])
p.close(); p.join(); del p
p = multiprocessing.Pool(NPROCESSORS)
p.map(functools.partial(filter_bg,
                        OUTPUT_DIR   = FILTERED_PATH,
                        BG_EXT_THRES = BG_EXT_THRES,
                        BG_EXT_ITER  = BG_EXT_ITER), all_files_channels)

################################################################################
p.close(); p.join(); del p
################################################################################
ffiles      = os.listdir(FILTERED_PATH)
ffiles      = map(lambda x: os.path.join(FILTERED_PATH, x), ffiles)
ffiles      = np.sort(ffiles)
tiff.imshow(tiff.imread(ffiles[0]))

print('All done.')


fluo = tiff.imread(ffiles[0])
fluo = fluo[:, 0,0]
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

''' Generate mesh '''
verts, faces, normals, values = measure.marching_cubes_lewiner(
    A.contour, 0, spacing=(2, 0.2516, 0.2516), step_size=1,
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
A.mesh = polygonPolyData
A.clean_mesh()
A.show_mesh(opacity=5)
A.compute_normals()

A.clean_mesh()
A.fill_holes(50.0)
#A.show_normals(opacity=.5)
A.smooth_mesh(iterations=0, relaxation_factor=.1)
A.quadric_decimation(dec=2000.0, method="npoints")
A.clean_mesh()
A.smooth_mesh(iterations=3000, relaxation_factor=.01, boundarySmoothing=True,
              featureEdgeSmoothing=True)



#from analyse import intensity_projection_series_all
#ffiles = os.listdir(FILTERED_PATH)
#ffiles = map(lambda x: os.path.join(FILTERED_PATH, x), ffiles)
