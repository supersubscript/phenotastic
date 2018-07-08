#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 12:53:20 2018

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

''' FILE INPUT '''
home = os.path.expanduser('~')

file_ = '/home/henrik/maxtest.tif'

f = fp.tiffload(file_)
meta = f.metadata
data = f.data.astype(np.float)
resolution = fp.get_resolution(f)

fluo = data[:, 0]
data = fluo

for ii in xrange(1):
    data = median_filter(data, size=1)
for ii in xrange(3):
    data = gaussian_filter(data, sigma=[1, 3, 3])

data = data.squeeze()
factor = .5
#data[data < np.mean(data)] = 0
contour = morphological_chan_vese(data, iterations=200,
                                  init_level_set=data > factor *
                                  np.mean(data),
                                  smoothing=2, lambda1=1, lambda2=1)


#contour = mp.fill_contour(contour)
#for ii in xrange(len(contour)):
#    contour[ii] = binary_fill_holes(contour[ii])

#A.contour = contour.copy()
#A.contour = A.contour.astype(np.float)
