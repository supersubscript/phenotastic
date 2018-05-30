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
import itertools
from skimage import measure
from vtk.util import numpy_support as nps
home = os.path.expanduser('~')
outFile = "rough_optimisation.dat"

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

#dir_ = os.path.abspath(os.path.join(home, "/home/henrik/data/from-benoit/parameter_optimisation_images/"))
dir_ = os.path.abspath(os.path.join(home, "/home3/From-Benoit/parameter_optimisation_images/"))

high_files = get_file_list_from_dir(dir_, pattern='*HighN*.tif')
low_files  = get_file_list_from_dir(dir_, pattern='*LowN*.tif')
target_files = get_file_list_from_dir(dir_, pattern='*data*.csv')

high_train, high_test = get_training_and_testing_sets(high_files)
low_train,  low_test  = get_training_and_testing_sets(low_files)

train = high_train + low_train
test  = high_test + low_test
train = map(lambda x: dir_ + '/' + x, train)
test  = map(lambda x: dir_ + '/' + x, test)
target_files =  map(lambda x: dir_ + '/' + x, target_files)
#import re
#digits = [re.findall(r'\d+', ii)[-1] for ii in test + train]
def stringSplitByNumbers(x):
  import re
  r = re.compile('(\d+)')
  l = r.split(x)
  return [int(y) if y.isdigit() else y for y in l]

all_ = sorted(train+test, key = stringSplitByNumbers)

### Parameters
# smoothing contour
# smoothing mesh
# relaxation mesh
# curvature type
# curvature smoothing distance
# boundary merge
# engulfing merge
# size removal

''' Read in target data '''
# NOTE: data_highN-8 possibly broken
# NOTE: data_highN-10 massively broken
data = pd.DataFrame()
for ii in target_files:
  print ii
  type_ = ii.split('_')[-1].split('-')[0]
  number = int(ii.split('_')[-1].split('-')[1].split('.csv')[0])
  newdata = pd.read_csv(ii).dropna()
  newdata['Type'] = type_
  newdata['Number'] = number
  data = data.append(newdata)
data = data[['Type', 'Number', 'Label', 'Center_X',
             'Center_Y', 'Center_Z', 'Value']]

def mp3d(comb):
    nPoints, contourSmoothing, meshSmoothing, meshRelaxation, curvatureSmoothingDist, boundaryMerge, engulfingMerge, sizeRemoval = comb # unpack

    nDomains = np.zeros(len(train)*2, dtype='int')
    ''' Simulation '''
    count = 0
    folder = '/'.join(train[0][:-4].split('/')[0:-1]) + '/contours'
    for file_ in train:
      A = ap.AutoPhenotype()
      fname = folder + '/' + file_[:-4].split('/')[-1] + '-' + str(contourSmoothing)
      A.load2(fname)
      A.quadric_decimation(dec=float(nPoints), method="npoints")
      A.clean_mesh()
      A.update_mesh()
      A.smooth_mesh(iterations=meshSmoothing, relaxation_factor=meshRelaxation, featureEdgeSmoothing=True, feature_angle=0)
  #    A.show_mesh()

      A.update_mesh()
      A.clean_mesh()

      dist_ = curvatureSmoothingDist
      curvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'max'],
                                operations=['-'], ignore_boundary=True)

      ''' Find domains '''
      pointData = init_pointData(A, curvs)
      pointData = boa.domains_from_curvature(pointData)
      boas, boasData = boa.get_boas(pointData)
      nDomains[count] = len(boas)


      pointData = boa.merge_boas_depth(pointData, threshold=boundaryMerge)
      pointData = boa.merge_boas_engulfing(A, pointData, threshold=engulfingMerge)
      pointData = boa.remove_boas_size(pointData, sizeRemoval,
                                       method="relative_all")
      boas, boasData = boa.get_boas(pointData)
      nDomains[count + 1] = len(boas)
      count += 2
      print "success!"

nPoints = [1000]
contourSmoothing = [10]
meshSmoothing = [50]
meshRelaxation = [0.1]
curvatureSmoothingDist = [1]
boundaryMerge = [0.001]
engulfingMerge = [0.69]
sizeRemoval1 = [0.01]


combs1 = list(itertools.product(
  nPoints,
  contourSmoothing,
  meshSmoothing,
  meshRelaxation,
  curvatureSmoothingDist,
  boundaryMerge,
  engulfingMerge,
  sizeRemoval1
))

from joblib import Parallel, delayed
import argparse
parser = argparse.ArgumentParser(prog='test')
parser.add_argument('-N', help='')
args = parser.parse_args()
if args.N == 0:
  Parallel(n_jobs=35)(delayed(mp3d)(comb=comb) for comb in combs1)

print "all done"
