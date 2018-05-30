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

os.chdir('/home/henrikahl/data/yi_curvature/tiffs')

#'''
#Define function which should be applied to all images
#'''
#def fit_single_paraboloid(filename):
#  try:
#    A = ap.AutoPhenotype()
#    A.data, _ = tiffread(filename)
#    print "Now running " + filename
#    A.contour_fit_threshold(threshold=.5, smooth_iterate=3)
#    A.mesh_conversion()
#    A.smooth_mesh(300, .5)
#    A.clean_mesh()
#    A.curvature_slice(0, 'mean')
#    A.clean_mesh()
#    A.feature_extraction(20)  # Only keep features > 20 % of pixels
#    A.sphere_fit()
#    A.sphere_evaluation()
#    A.paraboloid_fit_mersitem(weighted=False)
#    A.save(filename[:-4] + '_output')
#  except Exception:
#    print filename + "failed!"
#    pass
#
#
#"""
#FILE INPUT
#"""
folder   = '/home/henrikahl/data/yi_curvature/tiffs'
files    = os.listdir(folder)
files    = filter(lambda kk: '.tif' in kk, files)
inFiles  = [folder + "/" + files[ii] for ii in xrange(len(files))]
outFiles = [folder + "/" + files[ii][:-4] + "_output" for ii in xrange(len(files))]

#
##filename = '/home/henrikahl/home3/20171104-FM4-64/eli-1-8-g_sam.tif'
#
#p = Pool(7)
#p.map(fit_single_paraboloid, inFiles)



###--------------------------------------------------------------------------###
### Plot and stuff
###--------------------------------------------------------------------------###
outFiles = ['/home/henrik/plant2/00hrs_plant2_trim-acylYFP_hmin_2_asf_1_s_1.50_clean_3_output']
outFiles = ['/home/henrik/20171104-FM4-64-any1-1_sam_output']
outFiles = ['/home/henrik/20171104-FM4-64-any1-2_sam_output']
A = ap.AutoPhenotype()
A.load2(outFiles[0])
A.show_mesh()

