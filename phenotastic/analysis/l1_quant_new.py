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
from scipy.spatial import cKDTree

''' FILE INPUT '''
home = os.path.expanduser('~')

c1dir = '/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/c1/'
#c1dir = '/home/henrik/transfer/c1/'
#c3dir = '/home/henrik/transfer/c3/'

c3dir = '/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/c3/'
c1files = os.listdir(c1dir)
c1files = map(lambda x: os.path.join(c1dir, x), c1files)
c3files = os.listdir(c3dir)
c3files = map(lambda x: os.path.join(c3dir, x), c3files)
c1files.sort()
c3files.sort()

outdir = '/home/henrik/out_l1_quant4/'
outf = outdir + 'data.dat'

meshes = []
scalars = []

c1files = filter(lambda x: '24h' in x or '36h' in x or '48h' in x, c1files)
c3files = filter(lambda x: '24h' in x or '36h' in x or '48h' in x, c3files)

#c1files = c1files[2:]
#c3files = c3files[2:]
expressions=[]
for findex in xrange(0, len(c3files)):
    try:
        print('Running ' + c3files[findex])
        f = fp.tiffload(c3files[findex])
        meta = f.metadata
        data = f.data.astype(np.float)
        resolution = fp.get_resolution(f)
        fluo = data[:, 0]

        if resolution[0] < 2.:
            downfac = np.floor(2. / resolution[0]).astype(np.int)
            resolution[0] *= downfac
            fluo = fluo[::-1][::downfac][::-1]

        resolution[1:] = 0.25168

        ''' Create AutoPhenotype object to store the data in '''
        A = ap.AutoPhenotype()
        A.data = fluo.copy()
        A.data = A.data.astype(np.uint16)

        ''' Process data before creating contour. '''
        A.data[A.data < 3 * np.iinfo(np.uint16).max/np.iinfo(np.int8).max] = 0
#        A.data = clahe(A.data, np.array(A.data.shape) / 8, clip_limit=10)

        A.data = A.data.astype(np.float)
        A.data = A.data / np.max(A.data)

        for ii in xrange(1):
            A.data = median_filter(A.data, size=1)
        for ii in xrange(5):
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
        contour = morphological_chan_vese(A.data, iterations=1,
                                          init_level_set=A.data > factor *
                                          np.mean(A.data),
                                          smoothing=1, lambda1=1, lambda2=10)
        contour = mp.fill_contour(contour, fill_xy = False)
        for ii, slice_ in enumerate(contour):
            contour[ii] = binary_fill_holes(slice_)

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
        A.mesh = mp.remove_normals(A.mesh, threshold_angle=25, flip=False)
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

        wus = fp.tiffload(c1files[findex])
        wdata = wus.data
        wres = fp.get_resolution(wus)
        wres[1:] = 0.25168
        wdata = wdata.squeeze()

        if wres[0] < 2.:
            downfac = np.floor(2. / wres[0]).astype(np.int)
            wres[0] *= downfac
            wdata = wdata[::-1][::downfac][::-1]

        coords = pl.coord_array(wdata, xyzres/np.min(xyzres))
        vals = wdata.ravel().copy()
        vals[vals < 3 * np.mean(vals)] = 0

        coords = coords[vals > 0]
        vals = vals[vals > 0]

        ipd = vtk.vtkImplicitPolyDataDistance()
        ipd.SetInput(A.mesh)
        ipd.Modified()

        dists = np.zeros((len(coords),))
        pts = np.zeros((len(coords), 3))
        for ii in xrange(len(coords)):
            dists[ii] = ipd.EvaluateFunctionAndGetClosestPoint(coords[ii], pts[ii])

        if np.sum(nps.vtk_to_numpy(A.mesh.GetPointData().GetNormals()), axis=0)[0] < 0:
            dists *= -1
        filter_ = np.logical_and(dists > -1./np.min(xyzres) * 6, dists < 0)

        coords = coords[filter_]
        vals = vals[filter_]
        totvals = np.sum(vals)
        expressions.append(totvals)

        with open(outf, 'a') as of:
            of.writelines(c1files[findex] + "\t" + str(totvals) + "\n")

#
#        dists = dists[filter_]
#        tree = cKDTree(A.mesh.points)
#        closest = tree.query(coords, k=1)[1]
#
#        sumvals = np.zeros(A.mesh.points.shape[0])
#        for ii in xrange(len(coords)):
#            sumvals[closest[ii]] += vals[ii]
#
#
#        neighs = np.array([ap.get_connected_vertices(A.mesh, ii)
#                           for ii in xrange(A.mesh.npoints)])
#
#        sumvals_smooth = boa.filter_curvature(sumvals, neighs, np.mean, 1)

#        pobj = vi.PlotClass()
#        pobj.AddMesh(A.mesh)
#        pobj.AddPoints(coords, scalars=vals)
#        pobj.Plot()
#
#        meshes.append(A.mesh)
#        scalars.append(sumvals_smooth)



#        popt, _ = mp.fit_paraboloid_mesh(A.mesh)
#        A.mesh.RotateY(-90)
#        A.mesh.RotateX(-20)

#        pobj = vi.PlotClass()
#        pobj.AddMesh(A.mesh, scalars=sumvals_smooth)
#        pobj.Plot(in_background=False, autoclose=False, interactive=False)
#        pobj.TakeScreenShot(
#            outdir + '/figs/' + os.path.splitext(os.path.basename(c1files[findex]))[0] + '_surface.png')
#        pobj.Close()
#
#        A.mesh.RotateY(180)
#        pobj = vi.PlotClass()
#        pobj.AddMesh(A.mesh, scalars=sumvals_smooth)
#        pobj.Plot(in_background=False, autoclose=False, interactive=False)
#        pobj.TakeScreenShot(
#            outdir + '/figs/' + os.path.splitext(os.path.basename(c1files[findex]))[0] + '_surface2.png')
#        pobj.Close()

    except:
        continue
plants = np.unique(['-'.join(os.path.basename(c1files[ii]).split('-')[4:6]) for ii in xrange(len(c1files))])
c1files = c1files[:]
scalars = np.array(scalars)
plantmaxes = []
for ii in plants:
    filt = np.array(map(lambda x: ii in x, c1files))
    plantmax = np.hstack(scalars[filt]).max()
    plantmaxes.append(plantmax)
    print plantmax
    scalars[filt] /= plantmax
#
#for ii in xrange(len(meshes)):
#        meshes[ii].RotateY(-90)
#        meshes[ii].RotateX(-20)
##        meshes[ii].RotateZ(-180)
#
#        pobj = vi.PlotClass()
#        pobj.AddMesh(meshes[ii], scalars=scalars[ii])
#        pobj.Plot(in_background=False, autoclose=False, interactive=False)
#        pobj.TakeScreenShot(
#            outdir + '/figs/' + os.path.splitext(os.path.basename(c1files[ii]))[0] + '_surface.png', transparent_background=True)
#        pobj.Close()

#        A.mesh.RotateY(180)
#        pobj = vi.PlotClass()
#        pobj.AddMesh(meshes[ii], scalars=scalars[ii])
#        pobj.Plot(in_background=False, autoclose=False, interactive=False)
#        pobj.TakeScreenShot(
#            outdir + '/figs/' + os.path.splitext(os.path.basename(c1files[ii]))[0] + '_surface2.png', transparent_background=True)
#        pobj.Close()


#    A.mesh.Plot(scalars=sumvals, background='white')

#mesh_contour

    ###########################
#    wus = fp.tiffload(c1files[findex])
#    wdata = wus.data
#    wres = fp.get_resolution(wus)
#    wres[1:] = 0.25168
#    wdata = wdata.squeeze()
##
#    rescale_ = np.iinfo(wdata.dtype).max / np.iinfo(np.uint8).max
#    d = wdata.copy() / rescale_
#    d = np.ascontiguousarray(d)
#    d = d.astype(np.uint8)

##    image->SetOrigin(0,0,0);
##    image->SetScalarTypeToInt(); //I found the data type earlier in the program, this will change depending on your data type
##    image->SetNumberOfScalarComponents(0);
##    image->SetSpacing(xspacing, yspacing, zspacing); //the spacing between rows and columns of data and between slices
##    image->SetExtent(0, Rows-1, 0, Columns-1, 0, slices-1);
##    image->AllocateScalars();
#    #
#    imageImport = vtk.vtkImageImport()
##    imageImport.SetDataSpacing(wres[0], wres[1], wres[2])
#    imageImport.SetDataOrigin(0, 0, 0)
##    imageImport.SetWholeExtent(0, int((d.shape[0] - 1)*xyzres[0]), 0, int((d.shape[1] - 1)*xyzres[1]), 0, int((d.shape[2] - 1)*xyzres[2]))
#    imageImport.SetWholeExtent(0, int((d.shape[0] - 1)), 0, int((d.shape[1] - 1)), 0, int((d.shape[2] - 1)))
#    imageImport.SetDataExtentToWholeExtent()
#    imageImport.SetDataScalarTypeToUnsignedChar()
#    imageImport.SetNumberOfScalarComponents(1)
#    imageImport.SetImportVoidPointer(d)
#    imageImport.Update()
##
##    # Create the standard renderer, render window and interactor
#    ren = vtk.vtkRenderer()
#    renWin = vtk.vtkRenderWindow()
#    renWin.AddRenderer(ren)
#    iren = vtk.vtkRenderWindowInteractor()
#    iren.SetRenderWindow(renWin)
#
#    # Create transfer mapping scalar value to opacity
#    opacityTransferFunction = vtk.vtkPiecewiseFunction()
#    opacityTransferFunction.AddPoint(imageImport.GetOutput().GetScalarRange()[0], 0)
##    opacityTransferFunction.AddPoint(imageImport.GetOutput().GetScalarRange()[0] + 20, .1)
#    opacityTransferFunction.AddPoint(imageImport.GetOutput().GetScalarRange()[1], .7)
#
#    # Create transfer mapping scalar value to color
#    colorTransferFunction = vtk.vtkColorTransferFunction()
#    colorTransferFunction.AddRGBPoint(0.0, 1.0, 0.0, 0.0)
#    colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.0, 1.0)
#
#    # The property describes how the data will look
#    volumeProperty = vtk.vtkVolumeProperty()
#    volumeProperty.SetColor(colorTransferFunction)
#    volumeProperty.SetScalarOpacity(opacityTransferFunction)
#    volumeProperty.ShadeOn()
#    volumeProperty.SetInterpolationTypeToLinear()
##
##    # The mapper / ray cast function know how to render the data
#    volumeMapper = vtk.vtkSmartVolumeMapper() #vtk.vtkGPUVolumeRayCastMapper()
#    volumeMapper.SetBlendModeToComposite()
#    volumeMapper.SetInputData(imageImport.GetOutput())
##
##    # The volume holds the mapper and the property and
##    # can be used to position/orient the volume
#    volume = vtk.vtkVolume()
#    volume.SetMapper(volumeMapper)
#    volume.SetProperty(volumeProperty)
##
#    ren.AddVolume(volume)
#    ren.SetBackground(1, 1, 1)
#    renWin.SetSize(600, 600)
#    renWin.Render()
#
#    def CheckAbort(obj, event):
#        if obj.GetEventPending() != 0:
#            obj.SetAbortRender(1)
#
#    renWin.AddObserver("AbortCheckEvent", CheckAbort)
#
#    iren.Initialize()
#    renWin.Render()
#    iren.Start()