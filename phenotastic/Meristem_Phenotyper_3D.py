#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#######
#
# File author(s): Max BRAMBACH <max.brambach.0065@student.lu.se>
#                 Henrik ÅHL <henrik.aahl@slcu.cam.ac.uk>
# Copyright (c) 2017, Max Brambach, Henrik Åhl
# All rights reserved.
# * Redistribution and use in source and binary forms, with or without
# * modification, are not permitted.
#
############################################################################
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage.interpolation import zoom
from itertools import cycle
import numpy as np
import scipy.optimize as opt
from tissueviewer.mesh import tvMeshImageVTK
import vtk
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
import vtk.util.numpy_support as nps

import colorsys
import sys
import os
from tissueviewer.tvtiff import tiffread
from libtiff import TIFF
# import handy_functions as hf
from vtk.util import numpy_support
import copy
import domain_processing as boa
import networkx as nx
import tifffile as tiff
import vtkInterface as vi
from vtkInterface import PolyData
import misc
"""
===================
AutoPhenotype class
===================
"""

"""
Definition & Methods
===================
"""


class AutoPhenotype(object):
    """
    A python class for the automated extraction of SAM features in three
    dimensions from a stack of confocal microscope images.

    AutoPhenotype is a python class for the automated extraction of features of
    a shoot apical meristem (SAM) from a stack of confocal microscope images.
    Its main features are:

    * Extraction of the SAMs surface using active contour without edges (ACWE).
    * Separation of the SAMs features into primordiae and the meristem.
    * Evaluation of the features in terms of location, size and divergence angle.

    Parameters
    ----------

    Attributes
    ----------
    data : 3D intensity array, optional
        Image data e.g. from .tif file. Default = None.

    contour : 3D boolean array, optional
        Contour generated from data. Default = None.

    mesh : vi.PolyData, optional
        Mesh generated from contour. Can contain scalar values. Default = None.

    pdata : pd.DataFrame, optional
        Point data. Default = None.

    ddata : pd.DataFrame, optional
        Domain data. Default = None.

    Returns
    -------


    """

    def __init__(self, data=None, contour=None, mesh=None,
                 pdata=None, ddata=None):
        """Builder"""
        self.data = data
        self.contour = contour
        self.mesh = mesh
        self.pdata = pdata
        self.ddata = ddata

    def contour_fit_threshold(self, threshold=.8, smooth_iterate=3):
        """
        Generate contour fit via a threshold.

        Uses a threshold based approach to generate the initial model of
        self.data. Threshold default is half of the images mean intensity;
        can be adjusted. Subsequently, the contour is smoothed.

        Parameters
        ----------
        threshold :
             (Default value = .8)

        smooth_iterate :
             (Default value = 3)

        Returns
        -------


        """
        self.contour = np.array(self.data <
                                threshold * 0.5 * np.mean(self.data.flatten()),
                                dtype='int')
        self.smooth_contour(smooth_iterate)

    def step(self, weighting):
        """
        Perform a single step of the morphological Chan-Vese evolution.

        Implementation of the first two parts of Equation 32 in Marquez-Neila
        et al. 2014. (DOI: 10.1109/TPAMI.2013.106). The third part (smoothing)
        is implemented in the smooth_contour method.

        Parameters
        ----------
        weighting :

        Returns
        -------


        """
        if type(self.contour) == type(None):
            raise ValueError("the levelset function is not set (use .contour=)")
        inside = self.contour > 0
        outside = self.contour <= 0
        c0 = self.data[outside].sum() / float(outside.sum())
        c1 = self.data[inside].sum() / float(inside.sum())
        dres = np.array(np.gradient(self.contour))
        abs_dres = np.abs(dres).sum(0)
        aux = abs_dres * (weighting * (self.data - c1)**2 -
                          1. / float(weighting) * (self.data - c0)**2)
        self.contour[aux < 0] = 1
        self.contour[aux > 0] = 0

    def smooth_contour(self, iterate=1):
        """
        Smooth contour with ISoSI, SIoIS operators.

        Uses the SIoIS and ISoSI operator of Marquez-Neila et al. 2014
        (Section 3.5) to smooth the contour. At least one iteration after step
        is needed to get a good contour.

        Parameters
        ----------
        iterate :
             (Default value = 1)

        Returns
        -------

        """
        curvop = fcycle([SIoIS, ISoSI])
        res = self.contour
        for ii in xrange(iterate):
            res = curvop(res)
        self.contour = np.array(res, dtype='int16')

    def reduce(self, factor=2, spline=False):
        """
        Reduce the size of contour and data by the specified factor using
        slicing.

        The method deletes the unused planes (keeps only every nth - n=factor).
        Can also use spline filters for up and downsampling.

        Parameters
        ----------
        factor : int if spline = False, float else
            Specify the factor of reduction. (Default value = 2)


        spline : boolean
            * False (default): use numpy slicing to reduce number of planes.
            * True: use spline interpolation for reducing / enlarging data and
              contour.


        Returns
        -------
        No return :
            overwrite self.contour and self.data.

        """
        if spline == False:
            if type(self.data) != type(None):
                self.data = self.data[::factor, ::factor, ::factor]
            if type(self.contour) != type(None):
                self.contour = self.contour[::factor, ::factor, ::factor]
        else:
            if type(self.data) != type(None):
                self.data = zoom(self.data, zoom=1. / float(factor))
            if type(self.contour) != type(None):
                self.contour = zoom(self.contour, zoom=1. / float(factor))

    def contour_fit(self, weighting, iterate_smooth):
        """
        Run several full iterations of the morphological Chan-Vese method.

        Implementation of Equation 32 in Marquez-Neila et al. 2014.
        (DOI: 10.1109/TPAMI.2013.106).

        Parameters
        ----------
        iterations : int
            Number of times, the Chan-Vese Method is iterated.

        weighting : float
            Ratio of the weightings of the outside and inside contour.

        iterate_smooth : int
            Number of times, the smoothing operation is performed during one
            iteration.

        Returns
        -------
        No return :
            overwrite contour.

        """
        iterations = np.min(np.shape(self.data))
        self.set_contour_to_box()
        for i in range(iterations):
            self.step(weighting)
            self.smooth_contour(iterate_smooth)
            print("Contour fit: iteration %s/%s..." % (i + 1, iterations))

    def contour_fit_two_stage(self, iterations1, weighting1, iterate_smooth1,
                              iterations2, weighting2, iterate_smooth2,
                              zoom_factor):
        """
        Run a two staged contour fit. First fit is on down sampled image.

        The generated contour from the first fit is then up sampled and is used
        as initial contour for the regular fit.

        Parameters
        ----------
        iterations1,2 : int
            Number of times, the Chan-Vese Method is iterated.
            1 ->

        weighting1,2 : float
            Ratio of the weightings of the outside and inside contour.

        iterate_smooth1,2 : int
            Number of times, the smoothing operation is performed during one
            iteration.

        zoom_factor :

        Returns
        -------
        No return :
            overwrite contour.

        """
        data_large = self.data
        data_small = zoom(data_large, zoom=float(zoom_factor))
        self.data = data_small
        self.set_contour_to_box()
        self.contour_fit(iterations1, weighting1, iterate_smooth1)
        data_large_shape = np.shape(data_large)
        contour_large = zoom(self.contour, zoom=1. / float(zoom_factor) + .05)
        contour_large = contour_large[0:data_large_shape[0],
                                      0:data_large_shape[1],
                                      0:data_large_shape[2]]
        self.data = data_large
        self.contour = contour_large
        self.contour_fit(iterations2, weighting2, iterate_smooth2)

    def set_contour_to_box(self):
        """
        Set the contour attribute to a box.

        The box consists of ones on the surfaces except the edges and the first
        plane on the first axis (axis = 0). The rest is zeros.

        Parameters
        ----------

        Returns
        -------
        No return :
            Overwrites contour.

        """
        contour = getplanes(np.shape(self.data))
        contour = setedges(contour, 0)
        contour[0, :, :] = 0
        self.contour = contour

    def mesh_conversion(self):
        """
        Convert contour to mesh using marching cubes.

        Uses the marching cubes algorithm from vtk and the tvMeshImageVTK
        function is from tissuviewer.

        Parameters
        ----------

        Returns
        -------
        No return :
            Overwrites mesh.

        """
        fitval = 127  # basically 255/2
        fit = self.contour
        fit[fit == 0] = fitval
#         blockPrint()
        mesh = tvMeshImageVTK(fit, removeBoundaryCells=False, reduction=0,
                              smooth_steps=0)
#         enablePrint()
        self.mesh = mesh[fitval]

    def clean_mesh(self):
        """
        Extract largest connected set in self.mesh.

        Can be used to reduce residues of the contour fit inside the meristem.
        Works only if residues are not connected (share at least one point with)
        the meristems surface.

        Parameters
        ----------

        Returns
        -------
        No return :
            Overwrites mesh.

        """
        connect = vtk.vtkConnectivityFilter()

        if vtk.VTK_MAJOR_VERSION <= 5:
            connect.SetInput(self.mesh)
        else:
            connect.SetInputData(self.mesh)

        connect.SetExtractionModeToLargestRegion()
        connect.Update()
        geofilter = vtk.vtkGeometryFilter()

        if vtk.VTK_MAJOR_VERSION <= 5:
            geofilter.SetInput(connect.GetOutput())
        else:
            geofilter.SetInputData(connect.GetOutput())

        geofilter.Update()
        self.mesh = PolyData(geofilter.GetOutput())

    def smooth_mesh(self, iterations=500, relaxation_factor=.5,
                    featureEdgeSmoothing=False, boundarySmoothing=True,
                    feature_angle=30, edge_angle=15):
        """
        Smooth mesh.

        Uses vtk methods to clean and smooth the mesh. See documentation of
        vtkSmoothPolyData and vtkCleanPolyData for more info.

        Parameters
        ----------
        iterations :
             (Default value = 500)

        relaxation_factor :
             (Default value = .5)

        featureEdgeSmoothing :
             (Default value = False)

        boundarySmoothing :
             (Default value = True)

        feature_angle :
             (Default value = 30)

        edge_angle :
             (Default value = 15)

        Returns
        -------


        """
#        cleanPolyData = vtk.vtkCleanPolyData()
#
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        if vtk.VTK_MAJOR_VERSION <= 5:
            smoothFilter.SetInputConnection(self.mesh.GetOutputPort())
        else:
            smoothFilter.SetInputData(self.mesh)

        smoothFilter.SetNumberOfIterations(iterations)
        smoothFilter.SetRelaxationFactor(relaxation_factor)
        smoothFilter.SetFeatureAngle(feature_angle)
        smoothFilter.SetEdgeAngle(edge_angle)
        smoothFilter.SetOutputPointsPrecision(1)

        # Set special smoothing settings
        if featureEdgeSmoothing:
            smoothFilter.FeatureEdgeSmoothingOn()
        else:
            smoothFilter.FeatureEdgeSmoothingOff()
        if boundarySmoothing:
            smoothFilter.BoundarySmoothingOn()
        else:
            smoothFilter.BoundarySmoothingOff()
        smoothFilter.Update()
#
        self.mesh = PolyData(smoothFilter.GetOutput())

    def triangulate(self):
        """ """
        triangleFilter = vtk.vtkTriangleFilter()

        if vtk.VTK_MAJOR_VERSION <= 5:
            triangleFilter.SetInputConnection(self.mesh.GetProducerPort())
        else:
            triangleFilter.SetInputData(self.mesh)

        triangleFilter.Update()
        self.mesh = PolyData(triangleFilter.GetOutput())

    def decimate_mesh(self, fraction):
        """

        Parameters
        ----------
        fraction :


        Returns
        -------

        """
        decimate = vtk.vtkDecimatePro()

        if vtk.VTK_MAJOR_VERSION <= 5:
            decimate.SetInput(self.mesh)
        else:
            decimate.SetInputData(self.mesh)
        # //10% reduction (if there was 100 triangles, now there will be 90)
        decimate.SetTargetReduction(fraction)
        decimate.Update()

        self.mesh = PolyData(decimate.GetOutput())

    def compute_normals(self):
        """ """
        normalGenerator = vtk.vtkPolyDataNormals()

        if vtk.VTK_MAJOR_VERSION <= 5:
            normalGenerator.SetInputConnection(self.mesh.GetProducerPort())
        else:
            normalGenerator.SetInputData(self.mesh)

        normalGenerator.ComputePointNormalsOn()
        normalGenerator.ComputeCellNormalsOn()
        normalGenerator.SplittingOff()
        normalGenerator.FlipNormalsOn()
        normalGenerator.ConsistencyOn()
        normalGenerator.AutoOrientNormalsOn()
        normalGenerator.Update()
        self.mesh = PolyData(normalGenerator.GetOutput())


    def mesh_from_arrays(self, verts, faces):
        """

        Parameters
        ----------
        verts :

        faces :


        Returns
        -------

        """
        points = vtk.vtkPoints()
        points.SetData(nps.numpy_to_vtk(
            np.ascontiguousarray(verts), array_type=vtk.VTK_FLOAT, deep=True))
        nFaces = len(faces)
        faces = np.array([np.append(len(ii), ii) for ii in faces]).flatten()
        polygons = vtk.vtkCellArray()
        polygons.SetCells(nFaces, nps.numpy_to_vtk(
            faces, array_type=vtk.VTK_ID_TYPE))
        polygonPolyData = vtk.vtkPolyData()
        polygonPolyData.SetPoints(points)
        polygonPolyData.SetPolys(polygons)
        polygonPolyData.Update()

        self.mesh = PolyData(polygonPolyData)
        # TODO
        # for some reason this updates the mesh correctly
        self.smooth_mesh(iterations=0, relaxation_factor=.0)

    def curvature_slice(self, threshold=0., curv_types=['mean'], operations=[], lower=False):
        """
        Slice the mesh along (negative) curvature.

        Computes the curvature of the mesh and then uses a threshold filter to
        remove the parts with negative curvature.

        Parameters
        ----------
        threshold :
             (Default value = 0.)

        curv_types :
             (Default value = ['mean'])

        operations :
             (Default value = [])

        lower :
             (Default value = False)

        Returns
        -------
        No return :
            Overwrites mesh.

        """

        # Calculate curvatures
        self.calculate_curvatures(curv_types=curv_types, operations=operations)

        # Should we cut above or below this value?
        borders = vtk.vtkThreshold()
        if lower:
            borders.ThresholdByLower(threshold)
        else:
            borders.ThresholdByUpper(threshold)

        # Update mesh
        if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
            borders.SetInputConnection(self.mesh.GetProducerPort())
        else:
            borders.SetInputData(self.mesh)

        geofilter = vtk.vtkGeometryFilter()
        geofilter.SetInputConnection(borders.GetOutputPort())
        geofilter.Update()
        self.mesh = PolyData(geofilter.GetOutput())

    def slice_bottom(self, threshold=0., dim=0):
        """
        TODO

        Parameters
        ----------
        threshold :
             (Default value = 0.)

        dim :
             (Default value = 0)

        Returns
        -------

        """
        coords = nps.vtk_to_numpy(self.mesh.GetPoints().GetData())[:, dim]
        vtkCoords = nps.numpy_to_vtk(
            num_array=np.ascontiguousarray(coords), deep=True,
            array_type=vtk.VTK_DOUBLE)
        vtkCoords.SetName('Elevation')
        self.mesh.GetPointData().AddArray(vtkCoords)
        self.mesh.GetPointData().SetActiveScalars('Elevation')

        # Cut from the bottom up
        borders = vtk.vtkThreshold()
        minimumCoord = np.min(coords)
        borders.ThresholdByUpper(minimumCoord + threshold)

        # Update mesh
        geoFilter = vtk.vtkGeometryFilter()
        if vtk.VTK_MAJOR_VERSION <= 5:
            borders.SetInputConnection(self.mesh.GetProducerPort())
        else:
            borders.SetInputData(self.mesh)
        geoFilter.SetInputConnection(borders.GetOutputPort())
        geoFilter.Update()
        self.mesh = PolyData(geoFilter.GetOutput())

    def feature_extraction(self, min_percent=1.):
        """
        Extract the SAM features from the sliced mesh.

        Uses a vtk connectivity filter for the feature extraction.
        The points of the mesh are numbered. The method steps through the points
        (stepsize = total#ofPoints/res) and selects the connected surface in
        which the point lies. The selected surface is saved if:
            *Its #ofPoints is larger than stepsize
            *It has not already been selected
                (#ofPoints != #ofPoints(previous_iterations) )

        Parameters
        ----------
        min_percent :
             (Default value = 1.)

        Returns
        -------


        """
        connect = vtk.vtkPolyDataConnectivityFilter()
        connect.SetExtractionModeToSpecifiedRegions()
        connect.SetInput(self.mesh)
        connect.Update()
        num_regions = connect.GetNumberOfExtractedRegions()
        points = connect.GetInput().GetNumberOfCells()
        connect.Update()
        objects_vtk = []
        lastobj = []
        curvature = []
        thresh = min_percent / 100. * points
        connect.InitializeSpecifiedRegionList()
        for i in range(num_regions):
            connect.AddSpecifiedRegion(i)
            connect.Update()
            temp = []
            temp = connect.GetOutput()
            temp.Update()
            if temp.GetNumberOfCells() > thresh:
                connect2 = vtk.vtkConnectivityFilter()
                connect2.SetExtractionModeToLargestRegion()
                connect2.SetInput(temp)
                connect2.Update()
                geo = vtk.vtkGeometryFilter()
                geo.SetInput(connect2.GetOutput())
                geo.Update()
                objects_vtk.append(vtk.vtkPolyData())
                objects_vtk[-1].DeepCopy(geo.GetOutput())
                lastobj.append(objects_vtk[-1].GetNumberOfPoints())
                curv_temp = vtk_to_numpy(objects_vtk[-1].GetPointData(
                ).GetArray(self.curvature_type))
                curvature.append([abs(np.mean(curv_temp)),
                                  abs(np.std(curv_temp)),
                                  abs(np.max(curv_temp)),
                                  abs(np.min(curv_temp))])
            connect.DeleteSpecifiedRegion(i)

        self.features = objects_vtk
        npoints = pd.DataFrame(lastobj, columns=['points_in_feature'])
        self.results = self.results.append(npoints)
        curv = pd.DataFrame(np.array(curvature), columns=['mean_curvature',
                                                          'std_curvature',
                                                          'max_curvature',
                                                          'min_curvature'])
        self.results = pd.concat([self.results, curv], axis=1)

    def sphere_fit(self):
        """
        Fit spheres onto the features.

        Iterates over the vtkPolyData objects in features and performs a least
        square fit of a sphere to each feature.

        Parameters
        ----------

        Returns
        -------
        self.results : pd.DataFrame
            Four additional rows in self.results:
            * sphere_x_abs: absolute x-coordinate of the center of the
            fitted sphere
            * sphere_y_abs: absolute y-coordinate of the center of the
            fitted sphere
            * sphere_z_abs: absolute z-coordinate of the center of the
            fitted sphere
            Note: absolute means in units of the coordinate system used in
            self.features.

        """
        out = []
        print('Deprecated function.')
#        for i in range(np.shape(self.features)[0]):
#            out.append(fit_sphere(array_from_vtk_polydata(self.features[i])))
        fitval = pd.DataFrame(np.array(out), columns=['sphere_x_abs',
                                                      'sphere_y_abs',
                                                      'sphere_z_abs',
                                                      'sphere_radius',
                                                      'sphere_res_var'])
        self.results = pd.concat([self.results, fitval], axis=1)

    def sort_results(self, column='index', ascending=False, reset_index=False):
        """
        Sort results by specified column.

        The sorting direction can be adjusted.

        Parameters
        ----------
        column :
             (Default value = 'index')

        ascending :
             (Default value = False)

        reset_index :
             (Default value = False)

        Returns
        -------


        """
        if column == 'index':
            self.results.sort_index(ascending=True, inplace=True)
        else:
            self.results.sort_values(column, ascending=ascending, inplace=True)
        if reset_index == True:
            self.features = list(np.array(self.features)
                                 [self.results.index.values.tolist()])
            self.results.reset_index(inplace=True, drop=True)

    def sphere_evaluation(self):
        """
        Add several results to self.results.

        Added results are:
            *'sphere_x_rel': x-location of the primordium relative to the
                meristem.
            *'sphere_y_rel': y-location of the primordium relative to the
                meristem.
            *'sphere_z_rel': z-location of the primordium relative to the
                meristem.
            *'sphere_volume': Volume of the sphere
            *'sphere_R': Distance of the primordium to the meristem.
            *'sphere_angle_raw': Angle between the primordia in the y,z plane.
                Zero is chosen to be z = 0 and y > 0.
            *--results missing--
        Note: List of results is newly sorted in a way that the row with index
        0 is the meristem. The row label is changed accordingly.

        Parameters
        ----------

        Returns
        -------
        No return :
            Results are updated (see description).

        """
        if 'sphere_radius' not in self.results.columns:
            raise ValueError('Perform sphere fit first. Use self.sphere_fit()')
        self.sort_results('sphere_radius', reset_index=True)
        num_obj = self.results.shape[0]
        out = np.zeros((num_obj, 6))
        out[:, 0] = sphere_volume(self.results['sphere_radius'])  # volumes
        out[1:, 1] = self.results['sphere_x_abs'][1:] - self.results[
            'sphere_x_abs'][0]  # x relative to meristem
        out[1:, 2] = self.results['sphere_y_abs'][1:] - self.results[
            'sphere_y_abs'][0]  # y relative to meristem
        out[1:, 3] = self.results['sphere_z_abs'][1:] - self.results[
            'sphere_z_abs'][0]  # z relative to meristem
        out[1:, 4] = np.sqrt(out[1:, 1]**2. + out[1:, 2]
                             ** 2. + out[1:, 3]**2.)  # R
        out[1:, 5] = np.arctan2(out[1:, 2], out[1:, 3])  # theta'
        for i in range(1, num_obj):
            if out[i, 5] < 0:
                out[i, 5] = out[i, 5] + 2. * np.pi
        out[1:, 5] = out[1:, 5] / 2. / np.pi * 360.
        out_pd = pd.DataFrame(np.array(out), columns=['sphere_volume',
                                                      'sphere_x_rel',
                                                      'sphere_y_rel',
                                                      'sphere_z_resl',
                                                      'sphere_R',
                                                      'sphere_angle_raw'])
        self.results = pd.concat([self.results, out_pd], axis=1)

    def read_data(self, where):
        """

        Parameters
        ----------
        where :


        Returns
        -------

        """
        self.data, _ = tiffread(where)

    def save(self, where):
        """
        Saves the all data stored in an AutoPhenotype Object.

        Recognises whether or not data is available and saves only available.
        Note: it is advised to create a new folder for every save, since
        multiple files are created which always have the same name.
        The following files are saved (if avaliable):
            *data.tif : processed input data (e.g. reduced) as .tif stack
            *contour.tif : contour of data as .tif stack
            *mesh.vtp : mesh as vtk .vtp data (readable with
                vtk.vtkXMLPolyDataReader() )
            *featuresX.vtp : features in same format as mesh.vtp. Each feature
                is a separate file and X is a running number starting from 0.
            *results.csv : the results from e.g. the spherical fit as .csv
                data generated with pandas DataFrame.to_csv().

        Parameters
        ----------
        where :


        Returns
        -------


        """
        logfile = pd.DataFrame(np.zeros((1, 6)), columns=['data',
                                                          'contour',
                                                          'mesh',
                                                          'features',
                                                          'results',
                                                          'tags'])
        if not os.path.exists(where):  # checks if specified directory exists
            os.makedirs(where)         # creates one if not
        if type(self.data) != type(None):
            logfile['data'] = 1
            # TODO: Save metadata too
            tiff.imsave(self.data.astype(np.uint16), where + '/data.tif')
#            tiffsave(np.array(self.data, 'int16'), where + '/data.tif')
        if type(self.contour) != type(None):
            logfile['contour'] = 1
            # TODO: Save metadata too
            tiff.imsave(self.contour.astype(np.uint16), where + '/contour.tif')
#            tiffsave(np.array(self.contour, 'int16'), where + '/contour.tif')
        if type(self.mesh) != type(None):
            logfile['mesh'] = 1
            meshwriter = vtk.vtkXMLPolyDataWriter()
            meshwriter.SetInput(self.mesh)
            meshwriter.SetFileName(where + '/mesh.vtp')
            meshwriter.Write()
        if type(self.features) != type(None):
            os.makedirs(where + '/features')
            logfile['features'] = 1
            for i in range(np.shape(self.features)[0]):
                featurewriter = vtk.vtkXMLPolyDataWriter()
                featurewriter.SetInput(self.features[i])
                featurewriter.SetFileName(where + '/features/feature%s.vtp'
                                          % str(i))
                featurewriter.Write()
        if self.results.shape[0] != 0:
            logfile['results'] = 1
            self.results.to_csv(where + '/results.csv')
        logfile.to_csv(where + '/logfile.csv')

    def save_results(self, where):
        """
        Save results to specified directory.

        Results are saved as .csv file.

        Parameters
        ----------
        where :

        Returns
        -------


        """
        self.results.to_csv(where + '.csv')

    def reset_results(self, keep=['points_in_feature']):
        """
        Reset the result attribute.

        Results to be kept can be specified. Keeps points_in_feature by default.
        If all results should be deleted use keep = []

        Parameters
        ----------
        keep :
             (Default value = ['points_in_feature'])

        Returns
        -------


        """
        if type(keep) == type('string'):
            keep = [keep]
        self.results = self.results[keep]

    def clear(self, what):
        """
        Clear specified attributes.

        Attributes to be specified can be:
            *'all'
            *'data'
            *'tags'
            *'contour'
            *'mesh'
            *'features'
            *'results'

        Specified attributes are reset into initial condition.

        Parameters
        ----------
        what :


        Returns
        -------


        """
        if type(what) == type('string'):
            what = [what]
        if any(t == 'all' for t in what):
            self.data = None
            self.tags = None
            self.contour = None
            self.mesh = None
            self.features = None
            self.results = pd.DataFrame([], columns=['points_in_feature'])
        if any(t == 'data' for t in what):
            self.data = None
        if any(t == 'tags' for t in what):
            self.tags = None
        if any(t == 'contour' for t in what):
            self.contour = None
        if any(t == 'mesh' for t in what):
            self.mesh = None
        if any(t == 'features' for t in what):
            self.features = None
        if any(t == 'results' for t in what):
            self.results = pd.DataFrame([], columns=['points_in_feature'])

    def get_div_angle(self, sort_by='sphere_radius', sort_results=False):
        """
        Return divergence angles for angles sorted by sort_by.

        Can only be used after self.sphere_evaluation() has been used.
        Note: all angles must be in degree. Output is also in degree.

        Parameters
        ----------
        sort_by :
             (Default value = 'sphere_radius')

        sort_results :
             (Default value = False)

        Returns
        -------


        """
        results = self.results[['sphere_angle_raw', sort_by]]
        results = results.drop(0)
        results.sort_values(sort_by, ascending=False, inplace=True)
        if sort_results == True:
            self.sort_results(sort_by)
        return angle_difference(results['sphere_angle_raw'])

    def show_spheres(self, meristem_first=False, return_actors=False):
        """
        3D visualisation of the sphere fit.

        Uses vtk to show the fitted spheres.
        Color coding:
            *White: first entry in self.results
            *R->G->B->P: following results
        Note: This makes the script pause at the position of the call. Closing
        the render window lets the script continue.

        Parameters
        ----------
        meristem_first :
             (Default value = False)

        return_actors :
             (Default value = False)

        Returns
        -------
        type
            ------
            no return :
            Opens a render window.

        """
#        if meristem_first == True:
#            firstcolor = (1., 1., 1.)
#            lastcolor = ()
#        if meristem_first == False:
#            lastcolor = (1., 1., 1.)
#            firstcolor = ()
        spheres = self.results[['sphere_x_abs', 'sphere_y_abs', 'sphere_z_abs',
                                'sphere_radius']].as_matrix()
        sphereResolution = 50
        spheresSources = []
        for i in range(np.shape(spheres)[0]):
            spherevtk = vtk.vtkSphereSource()
            spherevtk.SetCenter(spheres[i, 0], spheres[i, 1], spheres[i, 2])
            spherevtk.SetRadius(spheres[i, 3])
            spherevtk.SetThetaResolution(sphereResolution)
            spherevtk.SetPhiResolution(sphereResolution)
            spherevtk.Update()
            spheresSources.append(PolyData(spherevtk.GetOutput()))
        print('Deprecated method.')
#        if return_actors == False:
#            view_polydata(spheresSources, firstcolor, lastcolor)
#        elif return_actors == True:
#            return view_polydata(spheresSources, firstcolor, lastcolor,
#                                 return_actors=True)

    # TODO:
    def show_paraboloid_and_mesh(self, p, sampleDim=(200, 200, 200),
                                 bounds=[-2000, 2000] * 3):
        """

        Parameters
        ----------
        p :

        sampleDim :
             (Default value = (200)

        Returns
        -------

        """
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
        tpoly.ClipPlane([self.GetBounds()[0] - 20, 0, 0], [1, 0, 0])

        pobj = vi.PlotClass()
        pobj.AddMesh(tpoly, opacity=.5, showedges=False, color='orange')
        pobj.AddMesh(self, opcaity=.9, color='green')
        pobj.Plot()

    def show_point_values(self, vals, stdevs=2, discrete=False,
                          return_actors=False, boaCoords=[], bg=[.1, .2, .3],
                          logScale=False, ruler=False):
        """

        Parameters
        ----------
        vals :

        stdevs :
             (Default value = 2)

        discrete :
             (Default value = False)

        return_actors :
             (Default value = False)

        boaCoords :
             (Default value = [])

        bg :

        logScale :
             (Default value = False)

        ruler :
             (Default value = False)

        Returns
        -------

        """
        # TODO: This function is a clusterfuck of a mess
        assert(isinstance(vals, pd.DataFrame))
#        vals = pd.DataFrame(np.array(pointData['domain']))
        output = vtk.vtkPolyData()
        output.ShallowCopy(self.mesh)

        if discrete:
            vals = pd.DataFrame(pd.Categorical(vals[0]).codes)
        if stdevs != "all" and not discrete:
            vals = misc.reject_outliers_2(vals, m=stdevs)

        if discrete:
            vals = pd.DataFrame(pd.Categorical(vals[0]).codes)
            scalarRange = [vals.min().values[0], vals.max().values[0]]
        scalarRange = [vals.min().values[0], vals.max().values[0]]

        if discrete:
            cols = misc.get_max_contrast_colours(n=len(np.unique(vals)))
            dctf = vtk.vtkDiscretizableColorTransferFunction()
            dctf.DiscretizeOn()
            dctf.SetRange(scalarRange[0], scalarRange[1])
            dctf.SetNumberOfValues(len(np.unique(vals)))
            for ii in xrange(len(cols)):
                dctf.AddRGBPoint(ii, cols[ii][0], cols[ii][1], cols[ii][2])
      #        dctf.SetNumberOfIndexedColors(len(np.unique(vals)))
            dctf.Build()

            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(len(np.unique(vals)))
            lut.SetRange(scalarRange)

            for ii in xrange(len(np.unique(vals))):
                rgb = [0, 0, 0]
                dctf.GetColor(float(ii) / len(np.unique(vals)), rgb)
                rgb = rgb + [1]
                print rgb
                lut.SetTableValue(ii, rgb)
            lut.Build()
        else:
            lut = vtk.vtkLookupTable()
            lutNum = 256
            lut.SetNumberOfTableValues(lutNum)

            if logScale:
                vals = vals + 0.000001
                lut.SetScaleToLog10()
                lut.SetRange(vals.min()[0], vals.max()[0])

            ctf = vtk.vtkColorTransferFunction()
            ctf.SetColorSpaceToDiverging()
            ctf.AddRGBPoint(0.0, 0, 0, 1.0)  # Blue
            ctf.AddRGBPoint(1.0, 1.0, 0, 0)  # Red
            for ii, ss in enumerate([float(xx) / float(lutNum) for xx in range(lutNum)]):
                cc = ctf.GetColor(ss)
                lut.SetTableValue(ii, cc[0], cc[1], cc[2], 1.0)

        # Add the array to the PointData and set is as the active one for
        # plotting.
        vtkPts = nps.numpy_to_vtk(vals, deep=True, array_type=vtk.VTK_FLOAT)
        vtkPts.SetName('Colors')
        output.GetPointData().AddArray(vtkPts)
        output.GetPointData().SetActiveScalars("Colors")

        # TODO: Add to returnActors too
        boaActors = []
        for ii in boaCoords:
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(ii)
            sphereSource.SetRadius(5.0)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            boaActors.append(actor)

        # Set colors for polydata object
        mapper = vtk.vtkPolyDataMapper()
        if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
            mapper.SetInputConnection(output.GetProducerPort())
        else:
            mapper.SetInputData(output)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(scalarRange)

        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Plot or return
        if return_actors:
            return actor
        acts = [actor]

        acts.extend(boaActors)
        misc.render_actors(acts, colorbar=True, ruler=ruler, bg=bg)

    # TODO:
    def show_normals(self, reverseNormals=False, onRatio=1, maxPoints=10000, scaleFactor=10, return_actors=False, opacity=1.0):
        """

        Parameters
        ----------
        reverseNormals :
             (Default value = False)

        onRatio :
             (Default value = 1)

        maxPoints :
             (Default value = 10000)

        scaleFactor :
             (Default value = 10)

        return_actors :
             (Default value = False)

        opacity :
             (Default value = 1.0)

        Returns
        -------

        """

        mesh = self.mesh

        # Generate normals
        normalGenerator = vtk.vtkPolyDataNormals()

        if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
            normalGenerator.SetInput(mesh)
        else:
            normalGenerator.SetInputData(mesh)

        normalGenerator.ComputePointNormalsOn()
        normalGenerator.ComputeCellNormalsOff()
        normalGenerator.SetSplitting(0)  # I want exactly one normal per vertex
        normalGenerator.GetOutput().GetPointData().SetActiveNormals('Normals')
        normalGenerator.Update()

        mapperNormals = vtk.vtkPolyDataMapper()
        mapperPoly = vtk.vtkPolyDataMapper()

        # Don't necessarily plot every single pointss
        maskPts = vtk.vtkMaskPoints()
        maskPts.SetOnRatio(onRatio)
        maskPts.RandomModeOn()
        maskPts.SetMaximumNumberOfPoints(maxPoints)
        maskPts.Update()

        reverse = vtk.vtkReverseSense()
        if reverseNormals:
            if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
                reverse.SetInput(mesh)
            else:
                reverse.SetInputData(mesh)

            reverse.ReverseCellsOn()
            reverse.ReverseNormalsOn()
            if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
                maskPts.SetInput(reverse.GetOutput())
            else:
                maskPts.SetInputData(reverse.GetOutput())
        else:
            if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
                maskPts.SetInput(mesh)
            else:
                maskPts.SetInputData(mesh)

        maskPts.Update()

        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(16)
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.1)
        arrow.Update()

        # use the output of vtkPolyDataNormals as input for the glyph3d
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetInputConnection(maskPts.GetOutputPort())
        glyph.SetVectorModeToUseNormal()
        glyph.SetScaleFactor(scaleFactor)
        glyph.SetColorModeToColorByVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.OrientOn()
        glyph.Update()

        mapperNormals.SetInputConnection(glyph.GetOutputPort())
        if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
            mapperPoly.SetInputConnection(mesh.GetProducerPort())
        else:
            mapperPoly.SetInputData(mesh)

        # Create actors
        actorNormals = vtk.vtkActor()
        actorNormals.SetMapper(mapperNormals)
        actorNormals.GetProperty().SetOpacity(opacity)
        actorPoly = vtk.vtkActor()
        actorPoly.SetMapper(mapperPoly)
        actorPoly.GetProperty().SetOpacity(opacity)

        # Create a renderer, render window, and interactor
        if return_actors:
            return [actorNormals, actorPoly]
        misc.render_actors([actorNormals, actorPoly])

    def show_spheres_and_features(self, return_actors=False):
        """"3D visualisation of the sphere fit.

        Uses vtk to show the fitted spheres.
        Color coding:
            * White: first entry in self.results
            * R->G->B->P: following results
        Note: This makes the script pause at the position of the call. Closing
        the render window lets the script continue.

        Parameters
        ----------
        return_actors :
             (Default value = False)

        Returns
        -------
        No return :
            Opens a render window.

        """
        features = self.features
        numel = len(features)
        spheres = self.results[['sphere_x_abs', 'sphere_y_abs', 'sphere_z_abs',
                                'sphere_radius']].as_matrix()
        sphereResolution = 50
        spheresSources = []
        for i in range(np.shape(spheres)[0]):
            spherevtk = vtk.vtkSphereSource()
            spherevtk.SetCenter(spheres[i, 0], spheres[i, 1], spheres[i, 2])
            spherevtk.SetRadius(spheres[i, 3])
            spherevtk.SetThetaResolution(sphereResolution)
            spherevtk.SetPhiResolution(sphereResolution)
            spherevtk.Update()
            spheresSources.append(spherevtk.GetOutput())
        sphereMappers = []
        featureMappers = []
        Actors = []
        render = vtk.vtkRenderer()
        s_colors = rgb_list(numel, (1., 1., 1.))
        f_colors = rgb_list(numel, (.7, .7, .7), v=.7)
        for i in range(numel):
            s_mapper = vtk.vtkPolyDataMapper()
            s_mapper.SetInput(spheresSources[i])
            s_mapper.ScalarVisibilityOff()
            s_mapper.Update()
            sphereMappers.append(s_mapper)
            f_mapper = vtk.vtkPolyDataMapper()
            f_mapper.SetInput(features[i])
            f_mapper.ScalarVisibilityOff()
            f_mapper.Update()
            featureMappers.append(f_mapper)
            s_actor = vtk.vtkActor()
            s_actor.SetMapper(sphereMappers[i])
            s_actor.GetProperty().SetColor(s_colors[i])
            Actors.append(s_actor)
            f_actor = vtk.vtkActor()
            f_actor.SetMapper(featureMappers[i])
            f_actor.GetProperty().SetColor(f_colors[i])
            Actors.append(f_actor)
            render.AddActor(Actors[-1])
            render.AddActor(Actors[-2])
        if return_actors == False:
            renderwindow = vtk.vtkRenderWindow()
            renderwindow.AddRenderer(render)
            renderwindow.SetSize(600, 600)
            interactrender = vtk.vtkRenderWindowInteractor()
            interactrender.SetRenderWindow(renderwindow)
            interactrender.Initialize()
            axes = vtk.vtkAxesActor()
            widget = vtk.vtkOrientationMarkerWidget()
            widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
            widget.SetOrientationMarker(axes)
            widget.SetInteractor(interactrender)
            widget.SetViewport(0.0, 0.0, 0.2, 0.2)
            widget.SetEnabled(1)
            widget.InteractiveOn()
            render.ResetCamera()
            renderwindow.Render()
            interactrender.Start()
        elif return_actors == True:
            return Actors

    def load_mesh(self, where):
        """
        TODO DOCS

        Parameters
        ----------
        where :

        Returns
        -------

        """
        reader = vtk.vtkPLYReader()
        reader.SetFileName(where)
        reader.Update()

        plyMapper = vtk.vtkPolyDataMapper()
        plyMapper.SetInputConnection(reader.GetOutputPort())

        plyActor = vtk.vtkActor()
        plyActor.SetMapper(plyMapper)
        self.mesh = plyActor.GetMapper().GetInput()

    def save_mesh_PLY(self, where):
        """
        TODO DOCS

        Parameters
        ----------
        where :


        Returns
        -------

        """
        plyWriter = vtk.vtkPLYWriter()
        plyWriter.SetFileName(where)
        plyWriter.SetInputConnection(self.mesh.GetProducerPort())
        plyWriter.Write()


"""
Functions
=========
"""


def setedges(array, value=0):
    """
    Set the edges of a 3D array to a specified value.

    Parameters
    ----------
    array : numpy array with shape() = (x,y,z)
        Array which edges are to be set to value.

    value : int, float
        Desired value for the edges of the array.

    Returns
    -------
    array : np.array
        Input array with edges set to value.

    """
    array[[0, 0, -1, -1], [0, -1, 0, -1], :] = value
    array[:, [0, 0, -1, -1], [0, -1, 0, -1]] = value
    array[[0, -1, 0, -1], :, [0, 0, -1, -1]] = value
    return array


def getedges(shape):
    """
    Create a 3D numpy array with zeros on the edges and ones else.

    Used to suppress the fitting of edges by the smooth() function
    (ISoSI operator mask).

    Parameters
    ----------
    shape : tuple with three components
        Shape of the returned array

    Returns
    -------
    array : np.array
        Three dimensional numpy array with zeros on the edges and ones else.

    """
    array = np.ones(shape)
    setedges(array)
    return array


def setplanes(array, value=0):
    """
    Set the surface of a 3D array to a specified value.

    Parameters
    ----------
    array : numpy array with shape() = (x,y,z)
        Array which surface are to be set to value.

    value : int, float
        Desired value for the surface of the array.

    Returns
    -------
    array : np.array
        Input array with surface set to value.

    """
    array[[0, -1], :, :] = value
    array[:, [0, -1], :] = value
    array[:, :, [0, -1]] = value


def getplanes(shape, invert=True):
    """
    Create a 3D numpy array with zeros on the surface and ones else or the
    other way around.

    Used as initial contour for the AutoPhenotype.step() method.

    Parameters
    ----------
    shape : tuple with three components
        Shape of the returned array

    invert : bool
        * True: Ones on surface, zeros else
        * False: Zeros on surface, ones else

    Returns
    -------
    array : np.array
        Three dimensional numpy array with zeros on the surface and ones else
        or the other way around.

    """
    if invert == True:
        array = np.zeros(shape)
        setplanes(array, value=1)
    if invert == False:
        array = np.ones(shape)
        setplanes(array, value=0)
    return array


class fcycle(object):
    """Call functions from the iterable each time it is called."""

    def __init__(self, iterable):
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)


def SI(u, iterate=1):
    """
    SI operator.

    Marquez-Neila et al. 2014. (DOI: 10.1109/TPAMI.2013.106)

    Parameters
    ----------
    u :

    iterate :
         (Default value = 1)

    Returns
    -------

    """
    P = [np.zeros((3, 3, 3)) for i in xrange(9)]
    P[0][:, :, 1] = 1
    P[1][:, 1, :] = 1
    P[2][1, :, :] = 1
    P[3][:, [0, 1, 2], [0, 1, 2]] = 1
    P[4][:, [0, 1, 2], [2, 1, 0]] = 1
    P[5][[0, 1, 2], :, [0, 1, 2]] = 1
    P[6][[0, 1, 2], :, [2, 1, 0]] = 1
    P[7][[0, 1, 2], [0, 1, 2], :] = 1
    P[8][[0, 1, 2], [2, 1, 0], :] = 1
    _aux = np.zeros((0))
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    for i in xrange(len(P)):
        _aux[i] = binary_erosion(u, P[i], iterations=iterate,
                                 mask=getedges(np.shape(u)))
    return _aux.max(0)


def IS(u, iterate=1):
    """
    IS operator.

    Marquez-Neila et al. 2014. (DOI: 10.1109/TPAMI.2013.106)

    Parameters
    ----------
    u :

    iterate :
         (Default value = 1)

    Returns
    -------

    """
    P = [np.zeros((3, 3, 3)) for i in xrange(9)]
    P[0][:, :, 1] = 1
    P[1][:, 1, :] = 1
    P[2][1, :, :] = 1
    P[3][:, [0, 1, 2], [0, 1, 2]] = 1
    P[4][:, [0, 1, 2], [2, 1, 0]] = 1
    P[5][[0, 1, 2], :, [0, 1, 2]] = 1
    P[6][[0, 1, 2], :, [2, 1, 0]] = 1
    P[7][[0, 1, 2], [0, 1, 2], :] = 1
    P[8][[0, 1, 2], [2, 1, 0], :] = 1
    _aux = np.zeros((0))
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    for i in xrange(len(P)):
        _aux[i] = binary_dilation(u, P[i], iterations=iterate,
                                  mask=getedges(np.shape(u)))
    return _aux.min(0)


def SIoIS(u):
    """
    SIoIS operator.

    Marquez-Neila et al. 2014. (DOI: 10.1109/TPAMI.2013.106)

    Parameters
    ----------
    u :


    Returns
    -------

    """
    return SI(IS(u))


def ISoSI(u):
    """
    ISoSI operator.

    Marquez-Neila et al. 2014. (DOI: 10.1109/TPAMI.2013.106)

    Parameters
    ----------
    u :


    Returns
    -------

    """
    return IS(SI(u))


def sort_a_along_b(b, a):
    """
    Return list 'a' sorted following the sorting of list 'b'.

    List 'b' is sorted from low to high values. Elements in 'a' follow the
    sorting in list 'b'.
    Example:  to array
        a = [p,c,e,s,e,i,l]; b = [5,2,7,6,1,4,3]
        -> sort_a_along_b(a,b) = [e,c,l,i,p,s,e];
        # and b would be [1,2,3,4,5,6,7]

    Parameters
    ----------
    b :

    a :


    Returns
    -------


    """
    return np.array(sorted(zip(a, b)))[:, 1]


def fit_sphere(data, init=[0, 0, 0, 10]):
    """
    Fit a sphere to specified data.

    Uses a least square fit for optimisation.
    Return coordinates of the sphere center and its radius as well as the
    residual variance of the fit.

    Parameters
    ----------
    data :

    init :

    Returns
    -------


    """
    def fitfunc(p, coords):
        x0, y0, z0, _ = p
        x, y, z = coords.T
        return ((x - x0)**2 + (y - y0)**2 + (z - z0)**2)

    def errfunc(p, x):
        return fitfunc(p, x) - p[3]**2.

    index = np.array(np.nonzero(data)).T
    p1, _ = opt.leastsq(errfunc, init, args=(index,))
    p1[3] = abs(p1[3])
    p1 = list(p1)
    p1.append(np.var(np.sqrt(np.square(index - p1[:3]).sum(1)) - p1[3]))

    return p1


def rot_coord(coord, angles, invert=False):
    """
    Rotate given coordinates by specified angles.

    Use rotation matrices to rotate a list of coordinates around the x, y, z axis
    by specified angles alpha, beta, gamma.

    Parameters
    ----------
    coord :

    angles :

    invert :
         (Default value = False)

    Returns
    -------


    """

    alpha, beta, gamma = angles
    xyz = np.zeros(np.shape(coord))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    if invert:
        R = np.linalg.inv(np.matmul(np.matmul(Rz, Ry), Rx))
    elif not invert:
        R = np.matmul(np.matmul(Rz, Ry), Rx)

    for ii in range(np.shape(coord)[0]):
        xyz[ii, :] = R.dot(np.array(coord[ii, :]))
    return xyz


def rot_matrix_44(angles, invert=False):
    """

    Parameters
    ----------
    angles :

    invert :
         (Default value = False)

    Returns
    -------

    """
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


def cart2sphere(xyz):
    """
    Convert cartesian coordinates into spherical coordinates.

    Convert a list of cartesian coordinates x, y, z to spherical coordinates
    r, theta, phi. theta is defined as 0 along z-axis.

    Parameters
    ----------
    xyz :

    Returns
    -------


    """
    rtp = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    rtp[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    rtp[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    rtp[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return rtp


def sphere2cart(rtp):
    """
    Convert spherical coordinates into cartesian coordinates.

    Convert a list of spherical coordinates r, theta, phi to cartesian coordinates
    x, y, z. Theta is defined as 0 along z-axis.

    Parameters
    ----------
    rtp :


    Returns
    -------


    """
    xyz = np.zeros(rtp.shape)
    xyz[:, 0] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
    xyz[:, 1] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
    xyz[:, 2] = rtp[:, 0] * np.cos(rtp[:, 1])
    return xyz


def rgb_list(N, firstcolor=(), lastcolor=(), s=1., v=1.):
    """
    Generate a list of N distinct RGB tuples.

    The first and last entry of the list can be specified. The list will still
    have N entries. The range of each tuple entry is between 0. and 1. The list
    goes from red over green to blue and purple.

    Parameters
    ----------
    N : int
        Number of colours.

    Returns
    -------
    RGB_tuples : tuple
        Tuples of RGB colours

    """
    if len(firstcolor) == 3 and len(lastcolor) == 3:
        HSV_tuples = [(float(x) / float(N), s, v) for x in range(N - 2)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        RGB_tuples.insert(0, firstcolor)
        RGB_tuples.append(lastcolor)
    elif len(firstcolor) == 3:
        HSV_tuples = [(float(x) / float(N), s, v) for x in range(N - 1)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        RGB_tuples.insert(0, firstcolor)
    elif len(lastcolor) == 3:
        HSV_tuples = [(float(x) / float(N), s, v) for x in range(N - 1)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        RGB_tuples.append(lastcolor)
    else:
        HSV_tuples = [(float(x) / float(N), s, v) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples


def sphere_volume(radius):
    """ Return the volume of a sphere of a given radius. """
    return 4. / 3. * np.pi * radius**3.


def angle_difference(array):
    """
    Return the differences between consecutive angles in an array.

    Computes both clockwise and counterclockwise angle differences.
    Angles need to be in degree.

    Parameters
    ----------
    array :

    Returns
    -------


    """
    clockwise = np.ediff1d(array) % 360.
    counterclockwise = np.abs(360. - clockwise)
    return clockwise, counterclockwise


def save_polydata_ply(what, where):
    """
    Save vtk.PolyData as .ply file.

    File location can be specified.

    Parameters
    ----------
    what : vi.PolyData
        PolyData to save.

    where : str
        Path for saving in.

    Returns
    -------
    No return :


    """
    meshwriter = vtk.vtkPLYWriter()
    meshwriter.SetInput(what)
    meshwriter.SetFileName(where + '.ply')
    meshwriter.Write()

def get_connected_vertices(mesh, seed, includeSelf=True):
    """

    Parameters
    ----------
    mesh :

    seed :

    includeSelf :
         (Default value = True)

    Returns
    -------

    """
    connectedVertices = []
    if includeSelf:
        connectedVertices.append(seed)

    cellIdList = vtk.vtkIdList()
    mesh.GetPointCells(seed, cellIdList)

    # Loop through each cell using the seed point
    for ii in xrange(cellIdList.GetNumberOfIds()):
        cell = mesh.GetCell(cellIdList.GetId(ii))		# get current cell

        # Loop through the edges of the point and add all points on these.
        for e in xrange(cell.GetNumberOfEdges()):
            pointIdList = cell.GetEdge(e).GetPointIds()

            # if one of the points on the edge are the vertex point, add the
            # other one
            if pointIdList.GetId(0) == seed:
                temp = pointIdList.GetId(1)
                connectedVertices.append(temp)
            elif pointIdList.GetId(1) == seed:
                temp = pointIdList.GetId(0)
                connectedVertices.append(temp)

    return np.unique(connectedVertices)


'''
Example
=======
'''
# if __name__ == "__main__":
#    A = AutoPhenotype()
#    A.data, _ = tiffread('test_images/meristem_test.tif')
#    A.contour_fit_threshold()
#    A.mesh_conversion()
#    A.clean_mesh()
#    A.smooth_mesh(300, .3)
#    A.curvature_slice()
#    A.feature_extraction()
#    A.sphere_fit()
#    A.sphere_evaluation()
#    A.paraboloid_fit_mersitem()
#    print A.results
#    A.show_spheres_and_features()
