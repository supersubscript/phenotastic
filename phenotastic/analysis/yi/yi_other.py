#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:14:17 2017

@author: henrikahl
"""

import os
import sys
import Meristem_Phenotyper_3D as ap
import handy_functions as hf
from tissueviewer.tvtiff import tiffread  # tiffsave
from multiprocessing import Pool

os.chdir('/home/henrikahl/projects/Meristem_Phenotyper_3D')

'''
Define function which should be applied to all images
'''
def fit_single_paraboloid(filename):
  try:
    A = ap.AutoPhenotype()
    A.data, _ = tiffread(filename)
    A.contour_fit_threshold(threshold=.5, smooth_iterate=3)
    A.mesh_conversion()
    A.smooth_mesh(300, .5)
    A.clean_mesh()
    A.curvature_slice(0, 'mean')
    A.clean_mesh()
    A.feature_extraction(20)  # Only keep features > 20 % of pixels
    A.sphere_fit()
    A.sphere_evaluation()
    A.paraboloid_fit_mersitem(weighted=False)
    A.save(filename[:-4] + '_output')
  except Exception:
    print filename + "failed!"
    sys.exc_clear()
    pass


"""
FILE INPUT
"""
folder = '/home/henrikahl/home3/files_15/'
filenames = os.listdir(folder)
files = []
for ii in xrange(len(filenames)):
    files.append(folder + '/' + filenames[ii])

p = Pool(3)
p.map(fit_single_paraboloid, files)


