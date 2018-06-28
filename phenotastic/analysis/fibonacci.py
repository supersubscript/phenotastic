# -*- coding: utf-8 -*-
#!/usr/bin/env python2
"""
Created on Sat Jun 16 20:25:03 2018

@author: henrik
"""

import os
import numpy as np
import vtk
import phenotastic.Meristem_Phenotyper_3D as ap
#from phenotastic import Meristem_Phenotyper_3D as ap
#import pandas as pd
import copy
#import handy_functions as hf
from skimage import measure
from vtk.util import numpy_support as nps
import tifffile as tiff
#from scipy.ndimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.segmentation import morphological_chan_vese
from scipy.ndimage.morphology import binary_fill_holes

#from scipy.ndimage.morphology import binary_fill_holes
import vtkInterface as vi
from skimage.exposure import equalize_hist
import phenotastic.domain_processing as boa
import phenotastic.plot as pl
import phenotastic.mesh_processing as mp
import phenotastic.file_processing as fp

#import phenotastic.misc
#from tissueviewer import fileProcessing as fp

''' FILE INPUT '''
home = os.path.expanduser('~')

#file_ = '/home/henrik/data/fibonacci/161010-COl0-LowN-24h-light-Soil-Sand/1-1-Soil/Col0-Seeds-LowN-24h-light-1-1-Soil-1.lsm'
#files = ['/home/henrik/data/fibonacci/161010-COl0-LowN-24h-light-Soil-Sand/1-1-Soil/Col0-Seeds-LowN-24h-light-1-1-Soil-1.lsm']

dir_ = '/home/henrik/data/fibonacci/161010-COl0-LowN-24h-light-Soil-Sand/1-1-Soil/'
#dir_ = '/home/henrik/data/fibonacci/160426-Col0-24h-light-Low-HighN/'
#dir_ = '/home/henrik/data/fibonacci/171110-pWUS-3X-VENUS-pCLV3-mCherry-Varying-Nutrients-Light/'
files = os.listdir(dir_)
files = map(lambda x: dir_ + x, files)

outdir = '/home/henrik/out_fib3'
m_outfile = outdir + '/meristem_data.dat'

with open(m_outfile, 'w') as f:
    f.writelines(np.array(['#index\t', 'fname\t', 'domain\t', 'dist_boundary\t', 'dist_com\t', 'angle\t', 'area\t', 'com_coords\t', 'ismeristem\n']))


for file_ in files:
#    file_ = files[0]
    f = fp.tiffload(file_)
    meta = f.metadata
    data = f.data.astype(np.float)
    resolution = fp.get_resolution(f)

    fluo = np.sum(data, axis=1)

    #from skimage.restoration import denoise_nl_means as dnl

    ''' Create AutoPhenotype object to store the data in '''
    A = ap.AutoPhenotype()
    A.data = fluo.copy()

    ''' Process data before creating contour. '''
#    maskfact = 1.0
#    A.data = equalize_hist(A.data, mask=A.data > maskfact * np.mean(A.data))
    #A.data = dnl(A.data, patch_distance=3, patch_size=2, multichannel=False)

    for ii in xrange(1):
        A.data = median_filter(A.data, size=1)
    for ii in xrange(3):
        A.data = gaussian_filter(A.data, sigma=[3/(resolution[0]/resolution[1]), 3, 3])

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
#    tiff.imshow(contour)

#    contour = mp.fill_contour(contour)
#    for ii in xrange(len(contour)):
#        contour[ii] = binary_fill_holes(contour[ii])
#    tiff.imshow(contour)
#    contour[0:10] = 0

    A.contour = contour.copy()
    A.contour = A.contour.astype(np.float)
################################################################################
    ''' Run MarchingCubes in skimage and convert to VTK format '''
    xyzres = resolution

    verts, faces, normals, values = measure.marching_cubes_lewiner(
        A.contour, 0, spacing=list(resolution/np.min(resolution)), step_size=1,
        allow_degenerate=False)
    faces = np.hstack(np.c_[[len(ii) for ii in faces], faces])
    surf = vi.PolyData(verts, faces)

    ''' Process mesh '''
    A.mesh = surf
    A.mesh.RotateY(-90)
    A.mesh = mp.remove_normals(A, threshold_angle=45, flip=False)
    A.mesh.RotateY(90)

    A.mesh.ExtractLargest()
    A.mesh.Clean()
    A.mesh.FillHoles(100)

################################################################################
    A.compute_normals()
    if np.sum(nps.vtk_to_numpy(A.mesh.GetPointData().GetNormals()), axis=0)[0] < 0:
        A.mesh.FlipNormals()
#    A.mesh.Plot()

    bounds = A.mesh.GetBounds()
    A.mesh.ClipPlane([np.ceil(bounds[0]), 0, 0], [0.0000001, 0, 0])
    A.mesh.ExtractLargest()
    A.mesh.FillHoles(100.0)
    A.compute_normals()

    A.mesh.ExtractLargest()
    A.mesh.FillHoles(100.0)
    A.compute_normals()
    A.mesh.Clean()
    A.mesh = A.mesh.Decimate(0.95, volume_preservation=True, normals=True)
    A.mesh.ExtractLargest()
    A.mesh.Clean()
    A.compute_normals()

    bounds = A.mesh.GetBounds()
    A.mesh.ClipPlane([20, 0, 0], [1,0,0])
    A.mesh.FillHoles(100.0)
    A.compute_normals()

    A.mesh.ExtractLargest()
    A.mesh.Clean()
#    A.mesh.Plot()

    A.mesh = mp.correct_bad_mesh(A.mesh)
    A.mesh.FillHoles(20.0)
    A.compute_normals()
    A.smooth_mesh(iterations=100, relaxation_factor=.01, boundarySmoothing=False,
                  featureEdgeSmoothing=False, feature_angle=45)
    A.mesh.ExtractLargest()
    A.mesh.Clean()

    # Sufficient loop to remesh
    while True:
        try:
            A.mesh = mp.remesh_decimate(A.mesh, iters=3)
            A.mesh = mp.remesh(A.mesh, A.mesh.points.shape[0])
        except:
            bounds = A.mesh.GetBounds()
            A.mesh.ClipPlane([bounds[0] + 1, 0, 0], [1,0,0])
            A.mesh.ExtractLargest()
            A.mesh.Clean()
            continue
        break

    A.mesh.Clean()
    A.mesh.ExtractLargest()
    A.compute_normals()
#    A.mesh = remove_normals(A, threshold=0, flip=False)

################################################################################
    # Check cumulative z(x)-directional normal orientation.
    if np.sum(nps.vtk_to_numpy(A.mesh.GetPointData().GetNormals()), axis=0)[0] < 0:
        A.mesh.FlipNormals()
    neighs = np.array([ap.get_connected_vertices(A.mesh, ii)
                       for ii in xrange(A.mesh.points.shape[0])])

    curvs = A.mesh.Curvature('mean')
    curvs = boa.set_boundary_curv(curvs, A.mesh, np.min(curvs))

    curvs = boa.filter_curvature(curvs, neighs, np.min, 1)
    curvs = boa.filter_curvature(curvs, neighs, np.mean, 10)

#    A.mesh.Plot(scalars=curvs)
#    A.mesh.Plot(background='w', scalars=curvs)

################################################################################
    ''' Create graphs '''
    pdata = boa.init_pointdata(A, curvs, neighs)

    ''' Identify BoAs'''
    pdata = boa.domains_from_curvature(pdata)
    boas, boasData = boa.get_boas(pdata)
    print boa.nboas(pdata)

    ''' Process boas '''
    safeCopy = copy.deepcopy(pdata)
    pdata = copy.deepcopy(safeCopy)
    boas, boasData = boa.get_boas(pdata)

    pdata = boa.merge_boas_depth(A, pdata, threshold=0.008)
    boas, boasData = boa.get_boas(pdata)
    print boa.nboas(pdata)

    pdata = boa.merge_boas_distance(pdata, boas, boasData, 15)
    boas, boasData = boa.get_boas(pdata)
    print boa.nboas(pdata)

    pdata = boa.merge_boas_engulfing(A, pdata, threshold=0.6)
    boas, boasData = boa.get_boas(pdata)
    print boa.nboas(pdata)

    pdata = boa.remove_boas_size(pdata, .05, method="relative_largest")
    boas, boasData = boa.get_boas(pdata)
    print boa.nboas(pdata)

    ''' Visualise '''
    print boa.nboas(pdata)
    print boa.boas_npoints(pdata)
    #A.show_curvatures(stdevs = "all", curvs = curvs)
    boas, boasData = boa.get_boas(pdata)
    boaCoords = np.array([tuple(ii) for ii in boasData[['z', 'y', 'x']].values])

#    pl.PlotPointData(A.mesh, pdata, 'domain',
#                     boaCoords=boaCoords, show_boundaries=True)

################################################################################
    ''' Export segmentation data '''
    meristem_index, _ = boa.define_meristem(
        A.mesh, pdata, method='central_mass', fluo=fluo)
    mpoly = boa.get_domain(A.mesh, pdata, meristem_index)

    # Find geometrical apex by fitting paraboloid
    popt, apexcoords = mp.paraboloid_fit_mersitem(mpoly)
    center_coord = mpoly.points[np.argmin(
        np.sqrt(np.sum((mpoly.points - apexcoords)**2, axis=1)))]

    try:
        # Extract domain data
        ddata = boa.extract_domaindata(pdata, A.mesh, apexcoords, meristem_index)
        pdata, ddata = boa.relabel_domains(pdata, ddata)

        # Merge based on domain angles
        angle_threshold = 14
        # TODO:
        pdata, ddata = boa.merge_boas_angle(pdata, ddata, A.mesh, angle_threshold, apexcoords)
        boas, boasData = boa.get_boas(pdata)
        boaCoords = np.array([tuple(ii) for ii in boasData[['z', 'y', 'x']].values])

#        pl.PlotPointData(A.mesh, pdata, 'domain',
#                         boaCoords=boaCoords, show_boundaries=True)

        res = np.array([360 - ii if np.abs(360 - ii - 137.5) < np.abs(ii - 137.5)
                        else ii for ii in np.abs(np.diff(ddata.angle.values))])
        print('Avg divergence angle: ' + str(np.mean(res[~np.isnan(res)])))

        ddata.insert(0, 'fname', [file_] * len(ddata))
        ddata.to_csv(m_outfile, sep='\t', header=False, mode='a')

        ''' Plot '''
        from vtkInterface.common import AxisRotation

        A.mesh.RotateY(-45)
        pobj = vi.PlotClass()
        pobj.AddMesh(A.mesh, scalars=pdata['domain'].values)
        pobj.AddPointLabels(
                AxisRotation(np.array(boaCoords), -45, axis='y'),
                np.array([str(ii) for ii in xrange(len(boaCoords))]),
                fontsize=30, pointcolor='w', textcolor='w')
        pobj.Plot(in_background=False, autoclose=False, interactive=False)
        pobj.TakeScreenShot(outdir + '/figs/' + os.path.splitext(os.path.basename(file_))[0] + '_top.png')
        pobj.Close()

        A.mesh.RotateY(90)
        pobj = vi.PlotClass()
        pobj.AddMesh(A.mesh, scalars=pdata['domain'].values)
        pobj.AddPointLabels(
                AxisRotation(np.array(boaCoords), 45, axis='y'),
                np.array([str(ii) for ii in xrange(len(boaCoords))]),
                fontsize=30, pointcolor='w', textcolor='w')
        pobj.Plot(in_background=False, autoclose=False, interactive=False)
        pobj.TakeScreenShot(outdir + '/figs/' + os.path.splitext(os.path.basename(file_))[0] + '_bottom.png', transparent_background=True)
        pobj.Close()

    except:
        pass


################################################################################
