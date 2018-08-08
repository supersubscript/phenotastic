# -*- coding: utf-8 -*-
#!/usr/bin/env python2
"""
Created on Sat Jun 16 20:25:03 2018

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

outdir = '/home/henrik/out_fib_comparison_corrected'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    os.mkdir(outdir + '/figs')
m_outfile = outdir + '/meristem_data.dat'

#with open(m_outfile, 'w') as f:
#    f.writelines(np.array(['#index\t', 'fname\t', 'domain\t', 'dist_boundary\t',
#                           'dist_com\t', 'angle\t', 'area\t', 'maxdist\t',
#                           'maxdist_xy\t', 'com_coords\t', 'ismeristem\n']))

for file_ in files:
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
#    curvs = boa.filter_curvature(curvs, neighs, np.min, 1)
#    curvs = boa.filter_curvature(curvs, neighs, np.mean, 1)

#    A.mesh.Plot(scalars=curvs, rng=0.01)

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

    pl.PlotPointData(A.mesh, pdata, 'domain',
                     boacoords=boacoords, show_boundaries=True)

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

    pl.PlotPointData(A.mesh, pdata, 'domain',
                     boacoords=boacoords, show_boundaries=True)
###############################################################################
    ''' Calculate stuff / write output '''
    from phenotastic.misc import divergence_angles
    res = divergence_angles(ddata.angle.values)
    print('Avg divergence angle: ' + str(np.mean(res[~np.isnan(res)])))

    ddata.insert(0, 'fname', [file_] * len(ddata))
    ddata.to_csv(m_outfile, sep='\t', header=False, mode='a')

    ''' Plot '''
    A.mesh.RotateY(-45)
    pobj = vi.PlotClass()
    pobj.AddMesh(A.mesh, scalars=pdata['domain'].values)
    pobj.AddPointLabels(
        AxisRotation(np.array(boacoords), -45, axis='y'),
        np.array([str(ii) for ii in xrange(len(boacoords))]),
        fontsize=30, pointcolor='w', textcolor='w')
    pobj.Plot(in_background=False, autoclose=False, interactive=False)
    pobj.TakeScreenShot(
        outdir + '/figs/' + os.path.splitext(os.path.basename(file_))[0] + '_top.png')
    pobj.Close()

    A.mesh.RotateY(90)
    pobj = vi.PlotClass()
    pobj.AddMesh(A.mesh, scalars=pdata['domain'].values)
    pobj.AddPointLabels(
        AxisRotation(np.array(boacoords), 45, axis='y'),
        np.array([str(ii) for ii in xrange(len(boacoords))]),
        fontsize=30, pointcolor='w', textcolor='w')
    pobj.Plot(in_background=False, autoclose=False, interactive=False)
    pobj.TakeScreenShot(
        outdir + '/figs/' + os.path.splitext(os.path.basename(file_))[0] + '_bottom.png')
    pobj.Close()

    ################################# PARAPLOTS


################################################################################
