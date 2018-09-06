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
from phenotastic.external.clahe import clahe
from phenotastic.misc import mkdir, listdir


''' FILE INPUT '''
home = os.path.expanduser('~')
dir_ = '/home/henrik/data/from-marcus/'

files = listdir(dir_, include='.tif')
files.sort()
#files = files[20:]

file_ = files[1]
import gc

def autocrop(arr, threshold=8e7, fct=np.sum):
    sumarr = np.max(arr, axis=1)

    cp = np.zeros((sumarr.ndim, 2), dtype=np.int)
    for ii in xrange(sumarr.ndim):
        summers = np.array([0,1,2])[np.array([0,1,2]) != ii]

        vals = fct(sumarr, axis=tuple(summers))
        first = next((e[0] for e in enumerate(vals) if e[1] > threshold), 0)
        last =  len(vals) - next((e[0] for e in enumerate(vals[::-1]) if e[1] > threshold), 0)

        cp[ii] = first, last

    return arr[cp[0, 0]:cp[0, 1], :, cp[1, 0]:cp[1, 1], cp[2, 0]:cp[2, 1]]

for file_ in files:
    try:
        gc.collect()
        f = fp.tiffload(file_)
        meta = f.metadata
#        resolution = np.array([meta['spacing'], meta['voxelsizey'], meta['voxelsizex']])
        data = f.data.astype(np.float)
        del f
        data = autocrop(data, 10e7, fct=np.sum)
        data = data[::-1]

        resolution = np.array((meta['spacing'], 0.3045961, 0.3045961))
#        if resolution[0] < 2.:
#            downfac = np.floor(2. / resolution[0]).astype(np.int)
#            resolution[0] *= downfac
#            data = data[::downfac]

#        fluo = data[:, 0]
        fluo = data.copy()
        fluo[:,1] /= np.max(fluo[:,1])
        fluo[:,0] /= np.max(fluo[:,0])

        fluo = np.max(data[:,:2], axis=1)

#        data = data[-200:]
#        fluo = fluo[-200:]

######

        ''' Create AutoPhenotype object to store the data in '''
        A = ap.AutoPhenotype()
        A.data = fluo
        A.data = A.data.astype(np.uint16)

        ''' Process data before creating contour. '''
#        A.data[A.data < 3 * np.iinfo(np.uint16).max / np.iinfo(np.uint8).max] = 0
#        A.data[A.data < 500] = 0
    #    A.data = clahe(A.data, np.array(A.data.shape) / 8, clip_limit=5)
        A.data = A.data.astype(np.float)
        A.data = A.data / np.max(A.data)

        for ii in xrange(1):
            A.data = median_filter(A.data, size=1)
#        for ii in xrange(1):
#            A.data = gaussian_filter(
#                A.data, sigma=[3. / resolution[0] * 0.2,
#                               3. / resolution[1] * 0.25,
#                               3. / resolution[2] * 0.25])

        ''' Smooth the data (to fill holes) and create a contour. lambda2 > lambda1:
          more variable on inside. Smoothing might have to be corrected for different
          xy-z dimensionality. Iterations should ideally be at least 10, smoothing
          around 4. '''
        A.data = (A.data - np.min(A.data)) / (np.max(A.data) - np.min(A.data))
        A.data = A.data * np.max(fluo)

        factor = .5
        contour = morphological_chan_vese(A.data, iterations=5,
                                          init_level_set=A.data > factor *
                                          np.mean(A.data),
                                          smoothing=1, lambda1=1, lambda2=1)

#        from skimage.morphology import binary_fill_holes
        from scipy.ndimage.morphology import binary_fill_holes
        for ii in xrange(contour.shape[0]):
            contour[ii] = binary_fill_holes(contour[ii])

#        contour = mp.fill_contour(contour, fill_xy=False)

#        pl.PlotImage(data, res=resolution)

    ################################################################################
        ''' Run MarchingCubes in skimage and convert to VTK format '''
#        contour = contour[:46]

        xyzres = resolution
        A.contour = contour

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            A.contour, 0, spacing=list(resolution / np.min(resolution)), step_size=1,
            allow_degenerate=False)
        faces = np.hstack(np.c_[np.full(faces.shape[0], 3), faces])
        A.mesh = vi.PolyData(verts, faces)

        del faces, verts, normals, values

        ''' Process mesh '''
        bounds = A.mesh.GetBounds()
        A.mesh.ClipPlane([np.ceil(bounds[0]), 0, 0], [(xyzres[0] + 0.0001), 0, 0])
        A.mesh.ClipPlane([np.floor(bounds[1]), 0, 0], [-(xyzres[0] + 0.0001), 0, 0])
        A.mesh.ClipPlane([0, np.ceil(bounds[2]), 0], [0, (xyzres[1] + 0.0001), 0])
        A.mesh.ClipPlane([0, np.floor(bounds[3]), 0], [0, -(xyzres[1] + 0.0001), 0])
        A.mesh.ClipPlane([0, 0, np.ceil(bounds[4])], [0, 0, (xyzres[2] + 0.0001)])
        A.mesh.ClipPlane([0, 0, np.floor(bounds[5])], [0, 0, -(xyzres[2] + 0.0001)])
#        A.mesh = mp.ECFT(A.mesh, 100.)
        A.mesh = mp.correct_bad_mesh(A.mesh)


#        A.mesh = mp.ECFT(A.mesh, 1000)
        bottom_cut = 130
        A.mesh = A.mesh.ClipPlane([bottom_cut, 0, 0], [1, 0, 0], inplace=False)
        A.mesh = mp.ECFT(A.mesh, 100)
        A.mesh = A.mesh.GenerateNormals(inplace=False)

#        A.mesh.RotateY(-90)
#        A.mesh = mp.remove_normals(A.mesh, threshold_angle=45, flip=False)
#        A.mesh.RotateY(90)
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
#        A.mesh = mp.drop_skirt(A.mesh, 1000)

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
#        A.mesh = mp.drop_skirt(A.mesh, 1000)
        A.mesh = mp.correct_bad_mesh(A.mesh)

        A.mesh = mp.remesh(A.mesh, A.mesh.npoints)
        A.mesh = A.mesh.GenerateNormals(inplace=False)

    ###############################################################################
#        if np.sum(nps.vtk_to_numpy(A.mesh.GetPointData().GetNormals()), axis=0)[0] < 0:
#            A.mesh.FlipNormals()
        neighs = np.array([ap.get_connected_vertices(A.mesh, ii)
                           for ii in xrange(A.mesh.npoints)])

        curvs = A.mesh.Curvature('mean')
        try:
            curvs = boa.set_boundary_curv(curvs, A.mesh, np.min(curvs))
        except:
            pass
        curvs = boa.filter_curvature(curvs, neighs, np.min, 1)
        curvs = boa.filter_curvature(curvs, neighs, np.mean, 50)
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

        pdata = boa.merge_boas_depth(A, pdata, threshold=0.001)
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

        d2 = data[:46, 1]
        md2 = data[:46, 2]

        wus = md2
        coords = pl.coord_array(wus, xyzres/np.min(xyzres))
        vals = wus.ravel().copy()
        vals[vals < 3 * np.mean(vals)] = 0
#        pobj = vi.PlotClass()
#        pobj.AddMesh(A.mesh)
#        pobj.AddPoints(coords, scalars=vals, psize=.1, opacity=.1)
#        pobj.Plot()

        coords = coords[vals > 0]
        vals = vals[vals > 0]

        from scipy.spatial import cKDTree
        #tree = cKDTree(A.mesh.points)
        #dists, idxs = tree.query(coords, k=1, distance_upper_bound=6)

        ### TODO:
        # 1. Close mesh (e.g. using 'repair')
        # 2. Get all values which are within mesh
        # 3. Get closest points on mesh to these values
        # 4. Sum up the intensity for these values and add them to the corresponding closest points
        # 5. Get angles and distance to apex for each of these points
        # 6. Do fourier analysis on this

        #pts = vi.utilities.MakeVTKPointsMesh(A.mesh.points)
        from pymeshfix import meshfix
        mf = meshfix.MeshFix(A.mesh)
        mf.Repair()
        rep_mesh = mf.mesh

        ipd = vtk.vtkImplicitPolyDataDistance()
        ipd.SetInput(A.mesh)
        ipd.Modified()

        dists = np.zeros((len(coords),))
        pts = np.zeros((len(coords), 3))
        for ii in xrange(len(coords)):
            dists[ii] = ipd.EvaluateFunctionAndGetClosestPoint(coords[ii], pts[ii])

#        filter_ = np.logical_and(dists > -6, dists < 0)
        filter_ = dists < 0
        #idxs = idxs[~np.isinf(dists)]
        coords = coords[filter_]
        vals = vals[filter_]
        dists = dists[filter_]
        #coords[idxs]
        pobj = vi.PlotClass()
        pobj.AddMesh(A.mesh)
        pobj.AddPoints(coords, scalars=vals, opacity=1)
        pobj.Plot()

        # Find the closest point on mesh to each point
        tree = cKDTree(A.mesh.points)
        closest = tree.query(coords, k=1)[1]

        sumvals = np.zeros(A.mesh.points.shape[0])
        for ii in xrange(len(coords)):
            sumvals[closest[ii]] += vals[ii]


        dom_vals = np.zeros((3, ))
        doms = pdata.domain.values
        npts = pdata.domain.value_counts().values
        for ii, val in enumerate(sumvals):
            dom_vals[doms[ii]] += val

        np.divide(a, npts)

#        A.mesh.Plot(scalars=sumvals, background='white')




    ################################################################################
#        ''' Find meristem / apex'''
#        meristem_index, _ = boa.define_meristem(
#            A.mesh, pdata, method='central_bounds', fluo=fluo)
#        mpoly = boa.get_domain(A.mesh, pdata, meristem_index)
#
#        # Find geometrical apex by fitting paraboloid
#        popt, apexcoords = mp.fit_paraboloid_mesh(mpoly)
##        center_coord = mpoly.points[np.argmin(
##            np.sqrt(np.sum((mpoly.points - apexcoords)**2, axis=1)))]
##        apexcoords = mpoly.CenterOfMass()
#
#    ################################################################################
#        ''' Merge more with new info '''
##        pdata = boa.merge_boas_disconnected(
##            A, pdata, meristem_index, threshold=.3, threshold2=.1)
##        boas, boasData = boa.get_boas(pdata)
##        print boa.nboas(pdata)
##
##        meristem_index, _ = boa.define_meristem(
##            A.mesh, pdata, method='central_mass', fluo=fluo)
##
#    ################################################################################
#        ''' Extract domain data '''
#        ddata = boa.extract_domaindata(pdata, A.mesh, apexcoords, meristem_index)
#        pdata, ddata = boa.relabel_domains(pdata, ddata, order='area')
#
#    ################################################################################
#        ''' Merge based on domain data '''
#    #    angle_threshold = 12
#    #    pdata, ddata = boa.merge_boas_angle(
#    #        pdata, ddata, A.mesh, angle_threshold, apexcoords)
#
#        pdata = boa.merge_boas_distance(pdata, boas, boasData, 15)
#        boas, boasData = boa.get_boas(pdata)
#        print boa.nboas(pdata)
#
#        pdata = boa.merge_boas_engulfing(A, pdata, threshold=0.6)
#        boas, boasData = boa.get_boas(pdata)
#        print boa.nboas(pdata)
#
#        meristem_index, _ = boa.define_meristem(
#            A.mesh, pdata, method='central_mass', fluo=fluo)
#
##        pdata = boa.merge_boas_disconnected(
##            A, pdata, meristem_index, threshold=.2, threshold2=.1)
##        boas, boasData = boa.get_boas(pdata)
##        print boa.nboas(pdata)
#
#        ddata = boa.extract_domaindata(pdata, A.mesh, apexcoords, meristem_index)
#        pdata, ddata = boa.relabel_domains(pdata, ddata, order='area')
#
#        boas, boasData = boa.get_boas(pdata)
#        boacoords = np.array([tuple(ii) for ii in boasData[['z', 'y', 'x']].values])
#
##        pl.PlotPointData(A.mesh, pdata, 'domain',
##                         boacoords=boacoords, show_boundaries=True)

        ########################################################################
        A.contour = contour
        A.pdata = pdata
        A.ddata = ddata
        A.data = data

