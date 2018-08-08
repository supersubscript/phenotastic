#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:43:58 2018

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
fib = '/home/henrik/data/yi/'
dirs = [fib + '1-2017mmdd',
        fib + '2-20170713',
        fib + '3-20170807',
        fib + '4-20170826']


files = []
for ii in dirs:
    ff = os.listdir(ii)
    ff = map(lambda x: os.path.join(ii, x), ff)
    files.extend(ff)

outdir = '/home/henrik/out_yi_curv/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    os.mkdir(outdir + '/figs')
m_outfile = outdir + '/meristem_data.dat'

with open(m_outfile, 'w') as f:
    f.writelines(np.array(['#index\t', 'fname\t', 'domain\t', 'dist_boundary\t',
                           'dist_com\t', 'angle\t', 'area\t', 'maxdist\t',
                           'maxdist_xy\t', 'com_coords\t', 'ismeristem\n']))

for file_ in files:
    try:
        f = fp.tiffload(file_)
        meta = f.metadata
        data = f.data.astype(np.float)
        resolution = fp.get_resolution(f)
        fluo = data[:, 0]

        if resolution[0] < 2e-6:
            downfac = np.floor(2e-6 / resolution[0]).astype(np.int)
            resolution[0] *= downfac
            fluo = fluo[::downfac]

        ind = fluo.shape[0] - next(x[0] for x in enumerate(fluo[:, fluo.shape[1]/2, fluo.shape[2]/2][::-1]) if x[1] > 25) - 1
        fluo = fluo[np.max([(ind - 20), 0]):np.min([(ind + 10), fluo.shape[0]])]

        ind11 = next(x[0] for x in enumerate(fluo[fluo.shape[0]/2, :, fluo.shape[2]/2]) if x[1] > 10)
        ind12 = fluo.shape[1] - next(x[0] for x in enumerate(fluo[fluo.shape[0]/2, :, fluo.shape[2]/2][::-1]) if x[1] > 10) - 1

        ind21 = next(x[0] for x in enumerate(fluo[fluo.shape[0]/2, fluo.shape[1]/2, :]) if x[1] > 10)
        ind22 = fluo.shape[2] - next(x[0] for x in enumerate(fluo[fluo.shape[0]/2, fluo.shape[1]/2, :][::-1]) if x[1] > 10) - 1

        fluo = fluo[:, ind11:ind12, ind21:ind22]

        ''' Create AutoPhenotype object to store the data in '''
        A = ap.AutoPhenotype()
        A.data = fluo.copy()
        A.data = A.data.astype(np.uint16)

        ''' Process data before creating contour. '''
        A.data[A.data < 3] = 0
    #    A.data = clahe(A.data, np.array(A.data.shape) / 8, clip_limit=5)
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

        bottom_cut = 1
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
    #    A.mesh = mp.drop_skirt(A.mesh, 1000)

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
    #    A.mesh = mp.drop_skirt(A.mesh, 1000)
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
        curvs = boa.filter_curvature(curvs, neighs, np.mean, 5)

    #    A.mesh.Plot(scalars=curvs)

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

        pdata = boa.merge_boas_depth(A, pdata, threshold=0.005)
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

    #    pl.PlotPointData(A.mesh, pdata, 'domain',
    #                     boacoords=boacoords, show_boundaries=True)

    ################################################################################
        ''' Find meristem / apex'''
        meristem_index, _ = boa.define_meristem(
            A.mesh, pdata, method='central_mass', fluo=fluo)
        mpoly = boa.get_domain(A.mesh, pdata, meristem_index)

        # Find geometrical apex by fitting paraboloid
        popt, apexcoords = mp.fit_paraboloid_mesh(mpoly)
        center_coord = mpoly.points[np.argmin(
            np.sqrt(np.sum((mpoly.points - apexcoords)**2, axis=1)))]
        apexcoords = mpoly.CenterOfMass()

    ################################################################################
        ''' Merge more with new info '''
        pdata = boa.merge_boas_disconnected(
            A, pdata, meristem_index, threshold=.3, threshold2=.1)
        boas, boasData = boa.get_boas(pdata)
        print boa.nboas(pdata)

        meristem_index, _ = boa.define_meristem(
            A.mesh, pdata, method='central_mass', fluo=fluo)

    ################################################################################
        ''' Extract domain data '''
        ddata = boa.extract_domaindata(pdata, A.mesh, apexcoords, meristem_index)
        pdata, ddata = boa.relabel_domains(pdata, ddata, order='area')

    ################################################################################
        ''' Merge based on domain data '''
    #    angle_threshold = 12
    #    pdata, ddata = boa.merge_boas_angle(
    #        pdata, ddata, A.mesh, angle_threshold, apexcoords)

        pdata = boa.merge_boas_distance(pdata, boas, boasData, 15)
        boas, boasData = boa.get_boas(pdata)
        print boa.nboas(pdata)

        pdata = boa.merge_boas_engulfing(A, pdata, threshold=0.6)
        boas, boasData = boa.get_boas(pdata)
        print boa.nboas(pdata)

        meristem_index, _ = boa.define_meristem(
            A.mesh, pdata, method='central_mass', fluo=fluo)

        pdata = boa.merge_boas_disconnected(
            A, pdata, meristem_index, threshold=.2, threshold2=.1)
        boas, boasData = boa.get_boas(pdata)
        print boa.nboas(pdata)

        ddata = boa.extract_domaindata(pdata, A.mesh, apexcoords, meristem_index)
        pdata, ddata = boa.relabel_domains(pdata, ddata, order='area')

        boas, boasData = boa.get_boas(pdata)
        boacoords = np.array([tuple(ii) for ii in boasData[['z', 'y', 'x']].values])

    #    pl.PlotPointData(A.mesh, pdata, 'domain',
    #                     boacoords=boacoords, show_boundaries=True)
        ###################### PARAPLOT

        def rot_matrix_44(angles, invert=False):
            alpha, beta, gamma = angles
            Rx = np.array([[1, 0, 0, 0],
                           [0, np.cos(alpha), -np.sin(alpha), 0],
                           [0, np.sin(alpha), np.cos(alpha), 0],
                           [0, 0, 0, 1]])
            Ry = np.array([[np.cos(beta), 0, np.sin(beta), 0],
                           [0, 1, 0, 0],
                           [-np.sin(beta), 0, np.cos(beta), 0],
                           [0, 0, 0, 1]])
            Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                           [np.sin(gamma), np.cos(gamma), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

            if invert == True:
                R = np.linalg.inv(np.matmul(np.matmul(Rz, Ry), Rx))
            elif invert == False:
                R = np.matmul(np.matmul(Rz, Ry), Rx)

            return R


        def PlotParaboloidSave(mesh, p, rot_ang, sampleDim=(200, 200, 200),
                           bounds=[-2000, 2000] * 3, ext = ''):
            p1, p2, p3, p4, p5, alpha, beta, gamma = p

            # Configure
            quadric = vtk.vtkQuadric()
            quadric.SetCoefficients(p1, p2, 0, 0, 0, 0, p3, p4, -1, p5)

            sample = vtk.vtkSampleFunction()
            sample.SetSampleDimensions(sampleDim)
            sample.SetImplicitFunction(quadric)
            sample.SetModelBounds(bounds)
            sample.Update()

            contour = vtk.vtkContourFilter()
            contour.SetInputData(sample.GetOutput())
            contour.Update()

            contourMapper = vtk.vtkPolyDataMapper()
            contourMapper.SetInputData(contour.GetOutput())
            contourActor = vtk.vtkActor()
            contourActor.SetMapper(contourMapper)

            rotMat = rot_matrix_44([alpha, beta, gamma], invert=True)
            trans = vtk.vtkMatrix4x4()
            for ii in xrange(0, rotMat.shape[0]):
                for jj in xrange(0, rotMat.shape[1]):
                    trans.SetElement(ii, jj, rotMat[ii][jj])

            transMat = vtk.vtkMatrixToHomogeneousTransform()
            transMat.SetInput(trans)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputData(contour.GetOutput())
            transformFilter.SetTransform(transMat)
            transformFilter.Update()
            tpoly = vi.PolyData(transformFilter.GetOutput())
            tpoly.ClipPlane([mesh.bounds[0] - 20, 0, 0], [1, 0, 0])

            mesh.RotateY(rot_ang)
            tpoly.RotateY(rot_ang)

            pobj = vi.PlotClass()
            pobj.AddMesh(mesh, scalars=pdata['domain'].values)
            pobj.AddPointLabels(
                AxisRotation(np.array(boacoords), -45, axis='y'),
                np.array([str(ii) for ii in xrange(len(boacoords))]),
                fontsize=30, pointcolor='w', textcolor='w')
            pobj.Plot(in_background=False, autoclose=False, interactive=False)
            pobj.TakeScreenShot(
                outdir + '/figs/' + os.path.splitext(os.path.basename(file_))[0] + '_segm_' + ext + '.png')
            pobj.Close()

            pobj = vi.PlotClass()
            pobj.AddMesh(tpoly, opacity=.5, showedges=False, color='orange')
            pobj.AddMesh(mesh, opcaity=.9, color='green')
            pobj.Plot(in_background=False, autoclose=False, interactive=False)
            pobj.TakeScreenShot(
                outdir + '/figs/' + os.path.splitext(os.path.basename(file_))[0] + '_para_' + ext + '.png')
            pobj.Close()

            mesh.RotateY(-rot_ang)
            tpoly.RotateY(-rot_ang)

        PlotParaboloidSave(boa.get_domain(A.mesh, pdata, 0), popt, -45, ext='bottom')
        PlotParaboloidSave(boa.get_domain(A.mesh, pdata, 0), popt, 45, ext='top')
    except:
        continue

