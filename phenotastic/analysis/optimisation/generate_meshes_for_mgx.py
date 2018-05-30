#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:16:25 2018

@author: henrikahl
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on 4 Aug 2017

@author: henrikahl
'''
import os
import numpy as np
import vtk
import Meristem_Phenotyper_3D as ap
import pandas as pd
import copy
import handy_functions as hf
import attractor_processing as boa
import networkx as nx
from skimage import measure
from vtk.util import numpy_support as nps
home = os.path.expanduser('~')

def init_pointData(A, curvs):
  nPoints = A.mesh.GetNumberOfPoints()
  coords = [A.mesh.GetPoint(ii) for ii in xrange(nPoints)]
  neighs = [ap.get_connected_vertices(A.mesh, ii) for ii in xrange(nPoints)]
  coords = pd.DataFrame(coords)
  neighs = pd.DataFrame(np.array(neighs))
  pointData = pd.concat([curvs, coords, neighs], axis=1)
  pointData.columns = ['curv', 'x', 'y', 'z', 'neighs']
  pointData['domain'] = np.NaN * nPoints
  return pointData

''' FILE INPUT '''

import random
random.seed(25)

def get_file_list_from_dir(datadir, pattern='*', fsuffix='', randomize=True):
    import fnmatch
    from random import shuffle
    all_files = os.listdir(os.path.abspath(datadir))
    all_files = fnmatch.filter(all_files, pattern)
    data_files = list(filter(lambda fname: fname.endswith(fsuffix), all_files))

    if randomize:
      shuffle(data_files)
    return data_files

def get_training_and_testing_sets(file_list, split=.3):
    import math
    split_index = int(math.floor(len(file_list) * split))
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

dir_ = os.path.abspath(os.path.join(home, "/home/henrik/data/from-benoit/parameter_optimisation_images/"))
dir_ = os.path.abspath(os.path.join(home, "/home/henrikahl/home3/From-Benoit/parameter_optimisation_images/"))

high_files = get_file_list_from_dir(dir_, pattern='*HighN*.tif')
low_files  = get_file_list_from_dir(dir_, pattern='*LowN*.tif')

high_train, high_test = get_training_and_testing_sets(high_files)
low_train,  low_test  = get_training_and_testing_sets(low_files)

train = high_train + low_train
test  = high_test + low_test
train = map(lambda x: dir_ + '/' + x, train)
test  = map(lambda x: dir_ + '/' + x, test)
#import re
#digits = [re.findall(r'\d+', ii)[-1] for ii in test + train]
def stringSplitByNumbers(x):
  import re
  r = re.compile('(\d+)')
  l = r.split(x)
  return [int(y) if y.isdigit() else y for y in l]

all_ = sorted(train+test, key = stringSplitByNumbers)

contourSmoothing  = 25
meshSmoothing     = 300
meshRelaxation    = 0.01
bottomSliceThresh = 10

for file_ in all_:
  print "Now processing " + file_
  A = ap.AutoPhenotype()
  A.read_data(file_)
  A.contour_fit_threshold(threshold=1.0, smooth_iterate=contourSmoothing)

  verts, faces, normals, values = measure.marching_cubes(
      A.contour, 0, spacing=(2.0, 1.0, 1.0))

  A.mesh_from_arrays(verts, faces)
  A.clean_mesh()
  A.slice_bottom(threshold=bottomSliceThresh , dim=0)
  A.clean_mesh()
  A.quadric_decimation(dec=2000.0, method="npoints")
  A.clean_mesh()
  A.smooth_mesh(iterations=meshSmoothing, relaxation_factor=meshRelaxation)
  A.update_mesh()
  A.clean_mesh()
  ap.save_polydata_ply(A.mesh, file_[:-4])