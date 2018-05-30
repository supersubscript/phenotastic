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
#from tissueviewer.mesh import tvMeshImageVTK
import vtk
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
import vtk.util.numpy_support as nps

import colorsys
import sys
import os
#from tissueviewer.tvtiff import tiffread, tiffsave
from libtiff import TIFF
import handy_functions as hf
from vtk.util import numpy_support
import copy
import domain_processing as boa
import networkx as nx
import tifffile as tiff
import vtkInterface as vi
from vtkInterface import PolyData
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
    """A python class for the automated extraction of SAM features in three
    dimensions from a stack of confocal microscope images.

    AutoPhenotype is a python class for the automated extraction of features of
    a shoot apical meristem (SAM) from a stack of confocal microscope images.
    Its main features are:

    *Extraction of the SAMs surface using active contour without edges (ACWE).
    *Separation of the SAMs features into primordiae and the meristem.
    *Evaluation of the features in terms of location, size and divergence angle.

    Attributes
    ----------
    data : 3D intensity array
        Image data e.g. from .tif file.

    tags : unused

    contour : 3D boolean array
        Contour generated from data.

    mesh : vtkPolyData
        Mesh generated from contour. Can contain curvature values.

    features : list of vtkPolyData
        The extracted features from mesh.

    results : panda.DataFrame
        The results of the evaluation operations (e.g. sphere fit). Also
        contains the number of points contained in each feature and its
        curvature.
        The number of rows is the number of primordia, whereas the  number of
        columns depends on the number of evaluation operations performed. New
        results are appended.
    """

    def __init__(self, data=None, tags=None, contour=None, mesh=None,
                 features=None):
        """Builder"""
        self.data = data
        self.tags = tags
        self.contour = contour
        self.mesh = mesh
        self.features = features
        self.results = pd.DataFrame([], columns=['points_in_feature'])

    def contour_fit_threshold(self, threshold=.8, smooth_iterate=3):
        """Generate contour fit via a threshold.

        Uses a threshold based approach to generate the initial model of
        self.data. Threshold default is half of the images mean intensity;
        can be adjusted.
        Subsequently, the contour is smoothed

        Parameters
        ----------
        threshold : float
            Sets the threshold for a voxel to be considered in- or outside the
            contour.
                *Inside: intensity(voxel) > mean_intensity/2*threshold
                *Outside: else

        iterate_smooth : int
            Number of times, the smoothing operation is performed.

        Return
        ------
        no return :
            Overwrites contour.
        """
        self.contour = np.array(self.data <
                                threshold * 0.5 * np.mean(self.data.flatten()),
                                dtype='int')
        self.smooth_contour(smooth_iterate)

    def step(self, weighting):
        """Perform a single step of the morphological Chan-Vese evolution.

        Implementation of the first two parts of Equation 32 in Marquez-Neila
        et al. 2014. (DOI: 10.1109/TPAMI.2013.106). The third part (smoothing)
        is implemented in the smooth_contour method.

        Parameters
        ----------
        weighting : float
            Ratio of the weightings of the inside to outside intensity.
            (lambda_1/lambda_2 in Equation 32, where lambda_2 is chosen to be 1

        Return
        ------
        no return :
            Overwrite self.contour.
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
        """Smooth contour with ISoSI, SIoIS operators.

        Uses the SIoIS and ISoSI operator of Marquez-Neila et al. 2014
        (Section 3.5) to smooth the contour. At least one iteration after step
        is needed to get a good contour.
        """
        curvop = fcycle([SIoIS, ISoSI])
        res = self.contour
        for ii in xrange(iterate):
            res = curvop(res)
        self.contour = np.array(res, dtype='int16')

    # TODO:
    def quadric_decimation(self, dec = 0.9, method="percentage"):
      decimate = vtk.vtkQuadricDecimation()
      if method == "percentage":
        decimate.SetTargetReduction(dec)
      elif method == "npoints":
        decimate.SetTargetReduction(1.0 - float(dec) / self.mesh.GetNumberOfPoints())

      decimate.VolumePreservationOn()
      decimate.NormalsAttributeOn()

      if vtk.VTK_MAJOR_VERSION < 6:
        decimate.SetInputConnection(self.mesh.GetProducerPort())
      else:
        decimate.SetInputData(self.mesh)

      decimate.Update()

      self.mesh = PolyData(decimate.GetOutput())

    # TODO:
    def reduce(self, factor=2, spline=False):
        """Reduce the size of contour and data by the specified factor using
        slicing.

        The method deletes the unused planes (keeps only every nth - n=factor).
        Can also use spline filters for up and downsampling.

        Parameter
        ---------
        factor : int if spline = False, float else
            Specify the factor of reduction.

        spline : boolean
            *if False (default): use numpy slicing to reduce number of planes.
            *if True: use spline interpolation for reducing / enlarging data and
              contour.

        Return
        ------
        no return :
            overwrite self.contour and self.data.
        """
        if spline == False:
            if type(self.data) != type(None):
                self.data = self.data[::factor, ::factor, ::factor]
            if type(self.contour) != type(None):
                self.contour = self.contour[::factor, ::factor, ::factor]
        if spline == True:
            if type(self.data) != type(None):
                self.data = zoom(self.data, zoom=1. / float(factor))
            if type(self.contour) != type(None):
                self.contour = zoom(self.contour, zoom=1. / float(factor))

    # TODO:
    def invert_normals(self):
      reverse = vtk.vtkReverseSense()
      if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
        reverse.SetInput(self.mesh)
      else:
        reverse.SetInputData(self.mesh)
      reverse.ReverseCellsOn()
      reverse.ReverseNormalsOn()
      self.mesh = PolyData(reverse.GetOutput())

    # TODO:
    def fill_holes(self, size=0.0):
        # Crashes atm. Might very well be current VTK version
        fill = vtk.vtkFillHolesFilter()

        if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
          fill.SetInput(self.mesh)
        else:
          fill.SetInputData(self.mesh)

        fill.SetHoleSize(size)

        # Make triangle window ordering consistent
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(fill.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()

        # Restore the original normals
        normals.GetOutput().GetPointData().SetNormals(self.mesh.GetPointData().GetNormals())
        normals.Update()

        self.mesh = PolyData(normals.GetOutput())

    # TODO:
    def update_mesh(self):
      if vtk.VTK_MAJOR_VERSION <= 5:
        self.mesh.Update()
      else:
        self.mesh.Modified()

    def contour_fit(self, weighting, iterate_smooth):
        """Run several full iterations of the morphological Chan-Vese method.

        Implementation of Equation 32 in Marquez-Neila et al. 2014.
        (DOI: 10.1109/TPAMI.2013.106).

        Parameter
        ---------
        iterations : int
            Number of times, the Chan-Vese Method is iterated.

        weighting : float
            Ratio of the weightings of the outside and inside contour.

        iterate_smooth : int
            Number of times, the smoothing operation is performed during one
            iteration.

        Return
        ------
        no return :
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
        """Run a two staged contour fit. First fit is on down sampled image.

        The generated contour from the first fit is then up sampled and is used
        as initial contour for the regular fit.

        Parameter
        ---------
        iterations1,2 : int
            Number of times, the Chan-Vese Method is iterated.
            1 ->

        weighting1,2 : float
            Ratio of the weightings of the outside and inside contour.

        iterate_smooth1,2 : int
            Number of times, the smoothing operation is performed during one
            iteration.

        Return
        ------
        no return :
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
        """Set the contour attribute to a box.

        The box consists of ones on the surfaces except the edges and the first
        plane on the first axis (axis = 0). The rest is zeros.

        Return
        ------
        no return :
            Overwrites contour.
        """
        contour = getplanes(np.shape(self.data))
        contour = setedges(contour, 0)
        contour[0, :, :] = 0
        self.contour = contour

    def mesh_conversion(self):
        """Convert contour to mesh using marching cubes.

        Uses the marching cubes algorithm from vtk and the tvMeshImageVTK
        function is from tissuviewer.

        Return
        ------
        no return :
            Overwrites mesh.
        """
        fitval = 127  # used to be 122, I think. Not sure what this sets
        fit = self.contour
        fit[fit == 0] = fitval
#         blockPrint()
        mesh = tvMeshImageVTK(fit, removeBoundaryCells=False, reduction=0,
                              smooth_steps=0)
#         enablePrint()
        self.mesh = mesh[fitval]

    def clean_mesh(self):
        """Extract largest connected set in self.mesh.

        Can be used to reduce residues of the contour fit inside the meristem.
        Works only if residues are not connected (share at least one point with)
        the meristems surface.

        Return
        ------
        no return :
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
        """Smooth mesh.

        Uses vtk methods to clean and smooth the mesh. See documentation of
        vtkSmoothPolyData and vtkCleanPolyData for more info.

        Parameters
        ----------
        iterations : int
            Number of iterations of the smooth algorithm.

        relaxation_factor: float
            Relaxation factor of the smooth algorithm.

        Return
        ------
        no return :
            Overwrites mesh.
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
      triangleFilter = vtk.vtkTriangleFilter()

      if vtk.VTK_MAJOR_VERSION <= 5:
        triangleFilter.SetInputConnection(self.mesh.GetProducerPort())
      else:
        triangleFilter.SetInputData(self.mesh)

      triangleFilter.Update()
      self.mesh = PolyData(triangleFilter.GetOutput())

    def decimate_mesh(self, fraction):
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

    # TODO:
    def calculate_curvatures(self, curv_types=['mean'], operations=[]):
      assert(len(operations) == len(curv_types) - 1)
      # TODO: Do in polish notation?
      # Calculate the different curvatures
      frames = []
      for ii in curv_types:
        if ii == 'mean':
          frames.append(curvature_mean(self.mesh))
        elif ii == 'max':
          frames.append(curvature_max(self.mesh))
        elif ii == 'min':
          frames.append(curvature_min(self.mesh))
        elif ii == 'gauss':
          frames.append(curvature_gauss(self.mesh))

      # Sum it up
      total = frames[0]
      for ii in xrange(len(operations)):
        total = eval("total" + operations[ii] + "(frames[ii + 1] + 1e-9)")

      curv_type_name = curv_types[0]
      for ii in xrange(len(operations)):
        curv_type_name += operations[ii] + curv_types[1:][ii]

      total = nps.numpy_to_vtk(total, deep=True, array_type=vtk.VTK_DOUBLE)
      total.SetName(curv_type_name)
      self.mesh.GetPointData().AddArray(total)
      self.mesh.GetPointData().SetActiveScalars(curv_type_name)
      self.curvature_type = curv_type_name
      self.update_mesh()

    def mesh_from_arrays(self, verts, faces):
        points = vtk.vtkPoints()
        points.SetData(nps.numpy_to_vtk(
            np.ascontiguousarray(verts), array_type=vtk.VTK_FLOAT, deep=True))
        nFaces = len(faces)
        faces = np.array([np.append(len(ii), ii) for ii in faces]).flatten()
        polygons = vtk.vtkCellArray()
        polygons.SetCells(nFaces, nps.numpy_to_vtk(faces, array_type=vtk.VTK_ID_TYPE))
        polygonPolyData = vtk.vtkPolyData()
        polygonPolyData.SetPoints(points)
        polygonPolyData.SetPolys(polygons)
        polygonPolyData.Update()


        self.mesh = PolyData(polygonPolyData)
        # TODO
        # for some reason this updates the mesh correctly
        self.smooth_mesh(iterations=0, relaxation_factor=.0)

    def curvature_slice(self, threshold=0., curv_types=['mean'], operations=[], lower=False):
        """Slice the mesh along (negative) curvature.

        Computes the curvature of the mesh and then uses a threshold filter to
        remove the parts with negative curvature.

        Return
        ------
        no return :
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
        ''' TODO '''
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
        """Extract the SAM features from the sliced mesh.

        Uses a vtk connectivity filter for the feature extraction.
        The points of the mesh are numbered. The method steps through the points
        (stepsize = total#ofPoints/res) and selects the connected surface in
        which the point lies. The selected surface is saved if:
            *Its #ofPoints is larger than stepsize
            *It has not already been selected
                (#ofPoints != #ofPoints(previous_iterations) )

        Parameters
        ----------
        min_percent : float (element of [0,100])
            Minimum percentage of cells / points in a primordium / SAM to be
            extracted. Connected regions with less cells / points are ignored.

        Return
        ------
        no return :
            Overwrites features.
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
        """Fit spheres onto the features.

        Iterates over the vtkPolyData objects in features and performs a least
        square fit of a sphere to each feature.

        Return
        ------
        self.results : pandas.DataFrame
            Four additional rows in self.results:
                *sphere_x_abs: absolute x-coordinate of the center of the
                    fitted sphere
                *sphere_y_abs: absolute y-coordinate of the center of the
                    fitted sphere
                *sphere_z_abs: absolute z-coordinate of the center of the
                    fitted sphere
            Note: absolute means in units of the coordinate system used in
            self.features.
        """
        out = []
        for i in range(np.shape(self.features)[0]):
            out.append(fit_sphere(array_from_vtk_polydata(self.features[i])))
        fitval = pd.DataFrame(np.array(out), columns=['sphere_x_abs',
                                                      'sphere_y_abs',
                                                      'sphere_z_abs',
                                                      'sphere_radius',
                                                      'sphere_res_var'])
        self.results = pd.concat([self.results, fitval], axis=1)

    def sort_results(self, column='index', ascending=False, reset_index=False):
        """Sort results by specified column.

        The sorting direction can be adjusted.

        Parameters
        ----------
        column : str
            Name of the column by which the array should be sorted.
            If column = 'index', the array will be sorted by its index

        ascending : bool
            Specifies the sorting direction:
                *False: high->low
                *True: low->high

        reset_index : bool
            Reset the row index after sorting the array. If True, the order of
            self.features is also changed accordingly.

        Return
        ------
        no return :
            Overwrites results
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
        """Add several results to self.results.

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
        Return
        ------
        no return :
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

    def paraboloid_fit_mersitem(self, weighted=False):
        """Fit the mersitem with a paraboloid.

        Uses the first entry of self.features as meristem.
        See fit_paraboloid() function for more info.

        Return
        ------
        no return :
            New results are added to self.results
            *para_p1 ... para_p5: Parameters of the paraboloid fit.
            *para_alpha,beta,gamma: Rotation of the paraboloid relative to
              image.
            *para_apex_x,y,z: Location of the paraboloids apex in coordinates
              of the image.
        """
        out = np.zeros([self.results.shape[0], 11])

        indices = np.array(np.nonzero(
            array_from_vtk_polydata(self.features[0]))).T

        if weighted:
            popt = fit_paraboloid_weighted(indices, )
        else:
            popt = fit_paraboloid(indices, )

        apex = get_paraboloid_apex(popt)
        out[0, :] = np.array(list(popt) + list(apex))
        fitval = pd.DataFrame(np.array(out), columns=['para_p1',
                                                      'para_p2',
                                                      'para_p3',
                                                      'para_p4',
                                                      'para_p5',
                                                      'para_alpha',
                                                      'para_beta',
                                                      'para_gamma',
                                                      'para_apex_x',
                                                      'para_apex_y',
                                                      'para_apex_z'])
        self.results = pd.concat([self.results, fitval], axis=1)

    def read_data(self, where):
      self.data, _ = tiffread(where)

    def save(self, where):
        """Saves the all data stored in an AutoPhenotype Object.

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
        where : string, path
            Path to the folder in which the data should be saved. If folder
            does not exist, it is created.
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
        """Save results to specified directory.

        Results are saved as .csv file.

        Parameters
        ----------
        where : string
            Path of the saved file. has to include the new filename wothout
            extension. .csv is automaticly added.
        """
        self.results.to_csv(where + '.csv')

    def load(self, where, what=None):
        """Load data to existing AutoPhenotype Object.

        What is loaded can be specified.

        Parameters
        ----------
        where : string, path
            Path to the folder from which the data should be loaded from.

        what : either None, logfile or list of strings
            *None: Logfile from specified folder is used.
            *logfile: No logfile is imported, instead the specified logfile is
                used. E.g. generated by create_logfile() function
            *list of strings: List of keywords specifying which data should be
                loaded. See logfile_from_str() for more information.
        Return
        ------
        no return :
            Attributes are overwritten.
        """
        if type(what) == type(None):
            logfile = pd.read_csv(where + '/logfile.csv')
        elif type(what) == type(create_logfile([1, 1, 1, 1, 1, 1])):
            logfile = what
        elif type(what) == type(['1', '2']):
            logfile = logfile_from_str(what)
        else:
            print ' - parameter what is unknown - '
            print 'If logfile.csv exists in directory (where) use what=None'
            print 'Else: either create logfile with create_logfile() function'
            print 'or specify files to load with strings:'
            print 'all, data, contour, mesh, features, results, tags'
            raise ValueError('parameter what is unknown')
        if logfile['data'][0] != 0:
            self.data, _ = tiffread(where + '/data.tif')
        if logfile['contour'][0] != 0:
            self.contour, _ = tiffread(where + '/contour.tif')
        if logfile['mesh'][0] != 0:
            meshreader = vtk.vtkXMLPolyDataReader()
            meshreader.SetFileName(where + '/mesh.vtp')
            meshreader.Update()
            self.mesh = PolyData(meshreader.GetOutput())
        if logfile['features'][0] != 0:
            self.features = []
            number_of_features = len(next(os.walk(where + '/features'))[2])
            for i in range(number_of_features):
                featurereader = vtk.vtkXMLPolyDataReader()
                featurereader.SetFileName(where + '/features/feature%s.vtp'
                                          % str(i))
                featurereader.Update()
                self.features.append(vtk.vtkPolyData())
                self.features[-1].DeepCopy(PolyData(featurereader.GetOutput()))
        if logfile['results'][0] != 0:
            self.results = pd.read_csv(where + '/results.csv')

    def reset_results(self, keep=['points_in_feature']):
        """Reset the result attribute.

        Results to be kept can be specified. Keeps points_in_feature by default.
        If all results should be deleted use keep = []

        Parameters
        ----------
        keep : list of strings
            List specifying the results to be kept.
            E.g. ['points_in_feature', 'sphere_radius']
            Note: Has to be list even if it only has one entry.
            Use [] for no results to be kept.

        Return
        ------
        no return :
            Overwrites self.results.
        """
        if type(keep) == type('string'):
            keep = [keep]
        self.results = self.results[keep]

    def clear(self, what):
        """Clear specified attributes.

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
        what : list of strings
            List of attributes to be cleared. See description.

        Return
        ------
        no return :
            Clears attributes.
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
        """Return divergence angles for angles sorted by sort_by.

        Can only be used after self.sphere_evaluation() has been used.
        Note: all angles must be in degree. Output is also in degree.

        Parameters
        ----------
        sort_by : str, index in self.results
            specify the order in which the organs are.
            E.g. sort by 'sphere_radius'
            Default is 'sphere_radius'.

        sort_results : bool
            Specify, if self.results should be sorted accordingly.
            Default is no sorting.

        Return
        ------
        clockwise : float list
            Difference between angles of the primordiae,
            clockwise orientation.

        counterclockwise : float list
            Difference between angles of the primordiae,
            counterclockwise orientation.
        """
        results = self.results[['sphere_angle_raw', sort_by]]
        results = results.drop(0)
        results.sort_values(sort_by, ascending=False, inplace=True)
        if sort_results == True:
            self.sort_results(sort_by)
        return angle_difference(results['sphere_angle_raw'])

    def show_spheres(self, meristem_first=False, return_actors=False):
        """"3D visualisation of the sphere fit.

        Uses vtk to show the fitted spheres.
        Color coding:
            *White: first entry in self.results
            *R->G->B->P: following results
        Note: This makes the script pause at the position of the call. Closing
        the render window lets the script continue.

        Return
        ------
        no return :
            Opens a render window.
        """
        if meristem_first == True:
            firstcolor = (1., 1., 1.)
            lastcolor = ()
        if meristem_first == False:
            lastcolor = (1., 1., 1.)
            firstcolor = ()
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
        if return_actors == False:
            view_polydata(spheresSources, firstcolor, lastcolor)
        elif return_actors == True:
            return view_polydata(spheresSources, firstcolor, lastcolor,
                                 return_actors=True)

    # TODO:
    def show_paraboloid_and_mesh(self, p, sampleDim=(200, 200, 200),
                             bounds=[-2000, 2000] * 3):
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
        tpoly.ClipPlane([self.GetBounds()[0] - 20, 0,0], [1,0,0])

        pobj = vi.PlotClass()
        pobj.AddMesh(tpoly, opacity=.5, showedges=False, color='orange')
        pobj.AddMesh(self, opcaity=.9, color='green')
        pobj.Plot()

    def show_point_values(self, vals, stdevs=2, discrete=False,
                          return_actors=False, boaCoords = [], bg = [.1,.2,.3],
                          logScale=False, ruler=False):
        # TODO: This function is a clusterfuck of a mess
        assert(isinstance(vals, pd.DataFrame))
#        vals = pd.DataFrame(np.array(pointData['domain']))
        output = vtk.vtkPolyData()
        output.ShallowCopy(self.mesh)

        if discrete:
            vals = pd.DataFrame(pd.Categorical(vals[0]).codes)
        if stdevs != "all" and not discrete:
            vals = hf.reject_outliers_2(vals, m=stdevs)

        if discrete:
            vals = pd.DataFrame(pd.Categorical(vals[0]).codes)
            scalarRange = [vals.min().values[0], vals.max().values[0]]
        scalarRange = [vals.min().values[0], vals.max().values[0]]

        if discrete:
          cols = hf.get_max_contrast_colours(n=len(np.unique(vals)))
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
              rgb = [0,0,0]
              dctf.GetColor(float(ii)/len(np.unique(vals)), rgb)
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
          ctf.AddRGBPoint(0.0, 0, 0, 1.0) # Blue
          ctf.AddRGBPoint(1.0, 1.0, 0, 0) # Red
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
        hf.render_actors(acts, colorbar=True, ruler=ruler, bg = bg)

    # TODO:
    def show_normals(self, reverseNormals=False, onRatio=1, maxPoints=10000, scaleFactor=10, return_actors=False, opacity=1.0):

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
        hf.render_actors([actorNormals, actorPoly])

    def show_curvatures(self, curv_types=['mean'], operations=[],
                        stdevs=2, numColors=2, curvs=None, normalise=False,
                        log_=False, return_actors = False):
#        self = A
#        curv_types=['mean']
#        operations=[]
#        stdevs=2
#        numColors=2
#        curvs=curvs
#        normalise=False
#        log_=False
#        return_actors = False

        # TODO: This function needs a major rewrite
        output = vtk.vtkPolyData()
        output.ShallowCopy(self.mesh)
        polyAlg = self.mesh.GetProducerPort()

        curvaturesFilter = vtk.vtkCurvatures()
        curvaturesFilter.SetInputConnection(polyAlg)

        # Get curvature values
        if not isinstance(curvs, pd.DataFrame):
            self.calculate_curvatures(curv_types=curv_types, operations=operations)
            curvVals = self.mesh.GetPointData().GetArray(self.curvature_type)
            curvVals = pd.DataFrame(nps.vtk_to_numpy(curvVals))
            curvs = copy.deepcopy(curvs)
        else:
            curvVals = copy.deepcopy(curvs)

        if stdevs != "all":
            curvVals = hf.reject_outliers_2(curvVals, m=stdevs)

        if normalise:
          min_ = curvs.min()
          max_ = curvs.max()
          curvVals = (curvVals - min_) / (max_ - min_)

        scalarRange = [curvVals.min().values[0], curvVals.max().values[0]]

        vtkarr = nps.numpy_to_vtk(
                num_array=curvVals.values.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        vtkarr.SetName(self.curvature_type)
        curvaturesFilter.GetOutput().GetPointData().AddArray(vtkarr)
        curvaturesFilter.GetOutput().GetPointData().SetActiveScalars(self.curvature_type)

        # Create the color map
        colorLookupTable = vtk.vtkLookupTable()
        colorLookupTable.SetTableRange(scalarRange[0], scalarRange[1])
#        colorLookupTable.SetNanColor(255, 255, 255, 0.0) # Set Nan-color to black
        if log_:
          colorLookupTable.SetScaleToLog10()
        colorLookupTable.Build()

        # Generate the colors for each point based on the color map
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        colors.SetLookupTable(colorLookupTable)
        m = curvs.median()[0]

        for ii in xrange(self.mesh.GetNumberOfPoints()):
            val = curvVals.iloc[ii, 0]

            # Color nan values black or white depending on whether they are too big or too small
            dcolor = np.zeros(3)
            if np.isnan(val):
              if curvs.iloc[ii, 0] > m:
                dcolor = np.array([0.,0.,0.])
              elif curvs.iloc[ii, 0] < m:
                dcolor = np.array([1.,1.,1.])
            else:
              colorLookupTable.GetColor(val, dcolor)

            color = np.zeros(3, dtype='int16')
            for jj in xrange(3):
                color[jj] = 255 * dcolor[jj] / 1.0

            colors.InsertNextTupleValue(color)

        output.GetPointData().AddArray(colors)
        output.GetPointData().SetActiveScalars("Colors")

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(output.GetProducerPort())
        mapper.SetLookupTable(colorLookupTable)

        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # TODO: Add colorbar as well
        if return_actors:
          return actor

        hf.render_actors([actor], colorbar=True)

    def show_spheres_and_features(self, return_actors=False):
        """"3D visualisation of the sphere fit.

        Uses vtk to show the fitted spheres.
        Color coding:
            *White: first entry in self.results
            *R->G->B->P: following results
        Note: This makes the script pause at the position of the call. Closing
        the render window lets the script continue.

        Return
        ------
        no return :
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

    def show_mesh(self, opacity=1.0):
        """"Visualise the mesh.

        Uses vtk to visualise the mesh.
        This can be done before or after the segmentation (using
        self.curvatures_slice() ).
        Note: This makes the script pause at the position of the call. Closing
        the render window lets the script continue.


        Return
        ------
        no return :
            Opens a render window.
        """
        view_polydata(self.mesh, (1., 1., 1.), (), opacity=opacity)

    def show_features(self):
        """"___.

        Uses vtk to visualise the mesh.
        This can be done before or after the segmentation (using
        self.curvatures_slice() ).
        Note: This makes the script pause at the position of the call. Closing
        the render window lets the script continue.


        Return
        ------
        no return :
            Opens a render window.
        """
        view_polydata(self.features, (1., 1., 1.), ())

    def load_mesh(self, where):
        ''' TODO DOCS '''
        reader = vtk.vtkPLYReader()
        reader.SetFileName(where)
        reader.Update()

        plyMapper = vtk.vtkPolyDataMapper()
        plyMapper.SetInputConnection(reader.GetOutputPort())

        plyActor = vtk.vtkActor()
        plyActor.SetMapper(plyMapper)
        self.mesh = plyActor.GetMapper().GetInput()

    def save_mesh_PLY(self, where):
        ''' TODO DOCS '''
        plyWriter = vtk.vtkPLYWriter()
        plyWriter.SetFileName(where)
        plyWriter.SetInputConnection(self.mesh.GetProducerPort())
        plyWriter.Write()

    def load2(self, where, what=None):
        """Load data to existing AutoPhenotype Object.

        What is loaded can be specified.

        Parameters
        ----------
        where : string, path
            Path to the folder from which the data should be loaded from.

        what : either None, logfile or list of strings
            *None: Logfile from specified folder is used.
            *logfile: No logfile is imported, instead the specified logfile is
                used. E.g. generated by create_logfile() function
            *list of strings: List of keywords specifying which data should be
                loaded. See logfile_from_str() for more information.
        Return
        ------
        no return :
            Attributes are overwritten.
        """
        if type(what) == type(None):
            logfile = pd.read_csv(where + '/logfile.csv')
        elif type(what) == type(create_logfile([1, 1, 1, 1, 1, 1])):
            logfile = what
        elif type(what) == type(['1', '2']):
            logfile = logfile_from_str(what)
        else:
            print ' - parameter what is unknown - '
            print 'If logfile.csv exists in directory (where) use what=None'
            print 'Else: either create logfile with create_logfile() function'
            print 'or specify files to load with strings:'
            print 'all, data, contour, mesh, features, results, tags'
            raise ValueError('parameter what is unknown')
        if logfile['data'][0] != 0:
            tiff = TIFF.open(where + "/data.tif", mode='r')
            ar = tiff.read_image()
            tiff.close()
            self.data = ar
#            self.data, _ = tiffread(where + '/data.tif')
        if logfile['contour'][0] != 0:
            tiff = TIFF.open(where + "/contour.tif", mode='r')
            ar = tiff.read_image()
            tiff.close()
            self.contour = ar
#            self.contour, _ = tiffread(where + '/contour.tif')
        if logfile['mesh'][0] != 0:
            meshreader = vtk.vtkXMLPolyDataReader()
            meshreader.SetFileName(where + '/mesh.vtp')
            meshreader.Update()
            self.mesh = PolyData(meshreader.GetOutput())
        if logfile['features'][0] != 0:
            self.features = []
            number_of_features = len(next(os.walk(where + '/features'))[2])
            for i in range(number_of_features):
                featurereader = vtk.vtkXMLPolyDataReader()
                featurereader.SetFileName(where + '/features/feature%s.vtp'
                                          % str(i))
                featurereader.Update()
                self.features.append(vtk.vtkPolyData())
                self.features[-1].DeepCopy(featurereader.GetOutput())
        if logfile['results'][0] != 0:
            self.results = pd.read_csv(where + '/results.csv')


"""
Functions
=========
"""


def setedges(array, value=0):
    """Set the edges of a 3D array to a specified value.

    Parameter
    ---------
    array : numpy array with shape() = (x,y,z)
        Array which edges are to be set to value.

    value : int, float
        Desired value for the edges of the array.

    Return:
    -------
    array : numpy array with shape() = (x,y,z)
        Input array with edges set to value.

    """
    array[[0, 0, -1, -1], [0, -1, 0, -1], :] = value
    array[:, [0, 0, -1, -1], [0, -1, 0, -1]] = value
    array[[0, -1, 0, -1], :, [0, 0, -1, -1]] = value
    return array



def getedges(shape):
    """Create a 3D numpy array with zeros on the edges and ones else.

    Used to suppress the fitting of edges by the smooth() function
    (ISoSI operator mask).

    Parameter
    ---------
    shape : tuple with three components
        Shape of the returned array

    Return:
    -------
    array : numpy array with shape() = shape
        Three dimensional numpy array with zeros on the edges and ones else.
    """
    array = np.ones(shape)
    setedges(array)
    return array


def setplanes(array, value=0):
    """Set the surface of a 3D array to a specified value.

    Parameter
    ---------
    array : numpy array with shape() = (x,y,z)
        Array which surface are to be set to value.

    value : int, float
        Desired value for the surface of the array.

    Return:
    -------
    array : numpy array with shape() = (x,y,z)
        Input array with surface set to value.

    """
    array[[0, -1], :, :] = value
    array[:, [0, -1], :] = value
    array[:, :, [0, -1]] = value


def getplanes(shape, invert=True):
    """Create a 3D numpy array with zeros on the surface and ones else or the
    other way around.

    Used as initial contour for the AutoPhenotype.step() method.

    Parameter
    ---------
    shape : tuple with three components
        Shape of the returned array

    invert : bool
        *True: Ones on surface, zeros else
        *False: Zeros on surface, ones else

    Return:
    -------
    array : numpy array with shape() = shape
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
    """SI operator.

    Marquez-Neila et al. 2014. (DOI: 10.1109/TPAMI.2013.106)
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
    """IS operator.

    Marquez-Neila et al. 2014. (DOI: 10.1109/TPAMI.2013.106)
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
    """SIoIS operator.

    Marquez-Neila et al. 2014. (DOI: 10.1109/TPAMI.2013.106)
    """
    return SI(IS(u))


def ISoSI(u):
    """ISoSI operator.

    Marquez-Neila et al. 2014. (DOI: 10.1109/TPAMI.2013.106)
    """
    return IS(SI(u))


def sort_a_along_b(b, a):
    """Return list 'a' sorted following the sorting of list 'b'.

    List 'b' is sorted from low to high values. Elements in 'a' follow the
    sorting in list 'b'.
    Example:  to array
        a = [p,c,e,s,e,i,l]; b = [5,2,7,6,1,4,3]
        -> sort_a_along_b(a,b) = [e,c,l,i,p,s,e];
        # and b would be [1,2,3,4,5,6,7]

    Parameters
    ----------
    a : list (same length as b)
        List to be sorted.

    b : list (same length as a)
        Reference list for sorting.

    Return
    ------
    a' : list
        List 'a' sorted with respect to 'b'.
    """
    return np.array(sorted(zip(a, b)))[:, 1]


def fit_sphere(data, init=[0, 0, 0, 10]):
    """Fit a sphere to specified data.

    Uses a least square fit for optimisation.
    Return coordinates of the sphere center and its radius as well as the
    residual variance of the fit.

    Parameters
    ----------
    data : 3D numpy array
        Data to be fit with a sphere

    init : list
        List of initial parameters:
        [x0, y0, z0, r0]

    Return
    ------
    parameter : list
        list of fitted parameters and residual variance:
        [x,y,z,r,res_var]
    """
    def fitfunc(p, coords):
        x0, y0, z0, _ = p
        x, y, z = coords.T
        return ((x - x0)**2 + (y - y0)**2 + (z - z0)**2)

    def errfunc(p, x): return fitfunc(p, x) - p[3]**2.
    index = np.array(np.nonzero(data)).T
    p1, _ = opt.leastsq(errfunc, init, args=(index,))
    p1[3] = abs(p1[3])
    p1 = list(p1)
    # res_var
    p1.append(np.var(np.sqrt(np.square(index - p1[:3]).sum(1)) - p1[3]))
    return p1


def fit_paraboloid(data, init=[1, 1, 1, 1, 1, 0, 0, 0]):
    """Fit a paraboloid to arbitrarily oriented 3D data.

    The paraboloid data can by oriented along an arbitrary axis. Not neccesary
    x,y,z. The function rotates the data points and returns the rotation angles
    along the x,y,z axis.
    Returns the parameters for a praboloid along the z-axis. The angles can be
    used to correct the paraboloid for rotation.
    Paraboloid equation : p1*x**2.+p2*y**2.+p3*x+p4*y+p5 = z

    Parameters
    ----------
    data : array or list of cartesian coordinates
        Cartesian coordinates of the data points.

    init : list of 8 scalars
        Initial parameter set.
        [p1, p2, p3, p4, p5, alpha, beta, gamma]
        *alpha: rotation around x axis
        *beta: rotation around y axis
        *gamma: rotation around z axis

    Return
    ------
    popt : list of 8 scalars
        Optimised parameters.
        [p1, p2, p3, p4, p5, alpha, beta, gamma]
        *alpha: rotation around x axis
        *beta: rotation around y axis
        *gamma: rotation around z axis
    """
    def errfunc(p, coord):
        p1, p2, p3, p4, p5, alpha, beta, gamma = p
        coord = rot_coord(coord, [alpha, beta, gamma])
        x, y, z = np.array(coord).T
        return abs(p1 * x**2. + p2 * y**2. + p3 * x + p4 * y + p5 - z)
#         return p1*x**2.+p2*y**2.+p3*x*y+p4*x+p5*y+p6 - z
    popt, _ = opt.leastsq(errfunc, init, args=(data,))
    return popt


def fit_paraboloid_weighted(data, init=[1, 1, 1, 1, 1, 0, 0, 0]):

    # TODO: write docs
    def errfunc(p, coord):
        p1, p2, p3, p4, p5, alpha, beta, gamma = p
        coord = rot_coord(coord, [alpha, beta, gamma])
        x, y, z = np.array(coord).T

        # Calculate the top of the paraboloid
        tx = -p3 / (2. * p1)
        ty = -p4 / (2. * p2)
        tz = p1 * tx**2. + p2 * ty**2. + p3 * tx + p4 * ty + p5

        return abs(p1 * x**2. + p2 * y**2. + p3 * x + p4 * y + p5 - z)

    popt, _ = opt.leastsq(errfunc, init, args=(data,))
    return popt


def get_paraboloid_apex(p):
    """Return the apex of a paraboloid.

    Use the return of fit_paraboloid() to compute the apex of the paraboloid.
    The return is in the coordinate system of the data, meaning that the
    coordinates have been corected for the rotation angles.

    Parameters
    ----------
    p : list of 8 scalars
        Optimised parameters.
        [p1, p2, p3, p4, p5, alpha, beta, gamma]
        *alpha: rotation around x axis
        *beta: rotation around y axis
        *gamma: rotation around z axis

    Return
    ------
    coord : list
        List of the apex' [x,y,z] coordinates.
    """
    p1, p2, p3, p4, p5, alpha, beta, gamma = p
    x = -p3 / (2. * p1)
    y = -p4 / (2. * p2)
    z = p1 * x**2. + p2 * y**2. + p3 * x + p4 * y + p5
    return rot_coord(np.array([[x, y, z], ]), [alpha, beta, gamma], True)[0]


def rot_coord(coord, angles, invert=False):
    """Rotate given coordinates by specified angles.

    Use rotation matrices to rotate a list of coordinates around the x,y,z axis
    by specified angles alpha,beta,gamma.

    Parameters
    ----------
    coord : list of coordinates
        List of catesian coordinates

    angles : list of three scalars
        List specifying the rotation angles. See description.

    invert : boolean
        True inverts the used rotation matrix. This can be used for undoing a
        rotation.

    Return
    ------
    rotated_coord : list of cordinates
        List of rotated cartesian coordinates
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

    if invert == True:
        R = np.linalg.inv(np.matmul(np.matmul(Rz, Ry), Rx))
#        R = np.linalg.inv(Rx.dot(Ry.dot(Rz)))
    elif invert == False:
        R = np.matmul(np.matmul(Rz, Ry), Rx)
#        R = Rx.dot(Ry.dot(Rz))

    for i in range(np.shape(coord)[0]):
        xyz[i, :] = R.dot(np.array(coord[i, :]))
    return xyz


def rot_matrix_44(angles, invert=False):
    # TODO: Write docs
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
#        R = np.linalg.inv(Rx.dot(Ry.dot(Rz)))
    elif invert == False:
        R = np.matmul(np.matmul(Rz, Ry), Rx)
#        R = Rx.dot(Ry.dot(Rz))

    return R


def cart2sphere(xyz):
    """Convert cartesian coordinates into spherical coordinates.

    Convert a list of cartesian coordinates x,y,z to spherical coordinates
    r,theta,phi. theta is defined as 0 along z-axis.

    Parameters
    ----------
    xyz : list
        List of cartesian coordinates

    Return
    ------
    rtp : list
        List of spherical coordinates
    """
    rtp = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    rtp[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    rtp[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    rtp[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return rtp


def sphere2cart(rtp):
    """Convert spherical coordinates into cartesian coordinates.

    Convert a list of spherical coordinates r,theta,phi to cartesian coordinates
    x,y,z. theta is defined as 0 along z-axis.

    Parameters
    ----------
    rtp : list
        List of spherical coordinates

    Return
    ------
    xyz : list
        List of cartesian coordinates
    """
    xyz = np.zeros(rtp.shape)
    xyz[:, 0] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
    xyz[:, 1] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
    xyz[:, 2] = rtp[:, 0] * np.cos(rtp[:, 1])
    return xyz


def array_from_vtk_polydata(poly, size=[]):
    """Create a boolean numpy array from vtkPolyData.

    The size of the created numpy array can be specified. It has to be equal or
    larger than the bounds of the polydata. Scalar values in the polyData are
    ignored. All nonzero values in poly are transformed into ones in the return.
    Note: works only in 3D.

    Parameters
    ----------
    poly : vtkPolyData (3D)
        polyData to be transformed.

    size : tuple / list of int
        Size of the created numpy array. By default set to the boundd of the
        polyData. e.g. (3,4,5)

    Return
    ------
    array : numpy array
        Boolean numpy array corresponding to the nonzero points in poly.
    """
    if np.shape(size) == np.shape([]):
        size = np.array(poly.GetPoints().GetBounds(), dtype='int')[1::2]
    indices = np.array(vtk_to_numpy(poly.GetPoints().GetData()), dtype='int')
    out = np.zeros(size)
    out[indices[:, 0] - 1, indices[:, 1] - 1, indices[:, 2] - 1] = 1
    return np.array(out)


def polydata_to_coord(poly):
    # Write all of the coordinates of the points in the vtkPolyData to the console.
    data = np.zeros((poly.GetNumberOfPoints(), 3))
    for ii in xrange(poly.GetNumberOfPoints()):
        data[ii, :] = poly.GetPoint(ii)

    return data


def coord_to_polydata(data):
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=data, deep=True, array_type=vtk.VTK_FLOAT)
    points = vtk.vtkPoints()
    points.SetData(vtk_data_array)
    return points
#    for ii in xrange(idNumPointsInFile):
#        value = array.GetValue(ii)

def view_polydata(poly, firstcolor=(), lastcolor=(), return_actors=False, opacity=1.0, bg=(0,0,0)):
    """Display vtkPolyData. Can show superposition of many vtkPolyData.

    If input is a list of vtkPolyData, displays all of them in one viewer.

    Parameters
    ----------
    poly : vtkPolyData (3D) / list of vtkPolyData (3D)
        polyData to be displayed. List or single polydata.

    Return
    ------
    no return :
        Opens render window.

    """
    if np.shape(poly) == ():
        numel = 1
        poly = [poly]
    else:
        numel = np.shape(poly)[0]
    if np.shape(firstcolor) != np.shape(()) and np.shape(
            lastcolor) != np.shape(()):
        colors = rgb_list(numel, firstcolor=firstcolor, lastcolor=lastcolor)
    elif np.shape(firstcolor) != np.shape(()):
        colors = rgb_list(numel, firstcolor=firstcolor)
    elif np.shape(lastcolor) != np.shape(()):
        colors = rgb_list(numel, lastcolor=lastcolor)
    else:
        colors = rgb_list(numel)
    Mappers = []
    Actors = []
    render = vtk.vtkRenderer()
    for i in range(numel):
        mapper = vtk.vtkPolyDataMapper()
        if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
          mapper.SetInput(poly[i])
        else:
          mapper.SetInputData(poly[i])
        mapper.ScalarVisibilityOff()
        mapper.Update()
        Mappers.append(mapper)
        actor = vtk.vtkActor()
        actor.SetMapper(Mappers[i])
        actor.GetProperty().SetColor(colors[i])
        actor.GetProperty().SetOpacity(opacity)
        Actors.append(actor)
        render.AddActor(Actors[i])
    if return_actors == False:
        render.SetBackground(bg)
        renderwindow = vtk.vtkRenderWindow()
        renderwindow.AddRenderer(render)
        renderwindow.SetSize(1200, 1200)
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


def rgb_list(N, firstcolor=(), lastcolor=(), s=1., v=1.):
    """Generate a list of N distinct RGB tuples.

    The first and last entry of the list can be specified. The list will still
    have N entries. The range of each tuple entry is between 0. and 1. The list
    goes from red over green to blue and purple.

    Parameters
    ----------
    N : int
        Number of returned RGB-tuples.

    firstcolor : list, three components
        First entry of the returned RGB-list.

    lastcolor : list, three components
        Last entry of the returned RGB-list.

    s : float between 0 and 1
        Saturation value of the list

    v : float between 0 and 1
        Value value of the list

    Return
    ------
    RGB-list : list of tuples with three components
        List with N RGB-tuples
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
    """Return the volume of a sphere.

    Is able to process numpy arrays.

    Parameters
    ----------
    R : ndarray of floats
        Radi of spheres.

    Return
    ------
    Volumes : ndarray of floats
        Volumes of spheres.
    """
    return 4. / 3. * np.pi * radius**3.


def angle_difference(array):
    """Return the differences between consecutive angles in an array.

    Computes both clockwise and counterclockwise angle differences.
    Angles need to be in degree.

    Parameters
    ----------
    array : list with n entries
        List of angles.

    Return
    ------
    angle differences : list with two entries each with n-1 entries
        *return[0]: clockwise angle differences
        *return[1]: counterclockwise angle differences
    """
    clockwise = np.ediff1d(array) % 360.
    counterclockwise = abs(360. - clockwise)
    return clockwise, counterclockwise


def create_logfile(logs, path=None):
    """Create a logfile for saving and loading AutoPhenotype data from boolean
    list.

    Return logfile and optionally save it as .csv file.

    Parameters
    ----------
    logs : list of 1,0
        Has to have six entries. 0: deactivated, 1: activated
            *Entry 0: data
            *Entry 1: contour
            *Entry 2: mesh
            *Entry 3: features
            *Entry 4: results
            *Entry 5: tags

    path = string
        If specified, the logfile is created at the specified location as .csv
        file. Path has to include file name.

    Return
    ------
    logfile : logfile
        Logfile which can be used in saving, loading AutoPhenotype data. Format
        is pandas.DataFrame()
    """
    logfile = pd.DataFrame([logs], columns=['data',
                                            'contour',
                                            'mesh',
                                            'features',
                                            'results',
                                            'tags'])
    if type(path) == type(None):
        return logfile
    if type(path) == type('string'):
        logfile.to_csv(path)
        return logfile


def logfile_from_str(what, path=None):
    """Create a logfile for saving and loading AutoPhenotype data from keywords

    Uses keywords to generate a logfile. Keywords can be:
        *'all': everything below
        *'data': input data as .tif
        *'contour': contour fit as .tif
        *'mesh': mesh as vtk data .vtp
        *'features': features as vtk data .vtp
        *'results': results as .csv

    Parameters
    ----------
    what : list of strings
        Use specified strings from description.
        Note: has to be list, even if only one keyword is specified.

    path = string
        If specified, the logfile is created at the specified location as .csv
        file. Path has to include file name.

    Return
    ------
    logfile : logfile
        Logfile which can be used in saving, loading AutoPhenotype data. Format
        is pandas.DataFrame()
    """
    logs = [0, 0, 0, 0, 0, 0]
    if any(t == 'all' for t in what):
        logs = [1, 1, 1, 1, 1, 1]
    if any(t == 'data' for t in what):
        logs[0] = 1
    if any(t == 'contour' for t in what):
        logs[1] = 1
    if any(t == 'mesh' for t in what):
        logs[2] = 1
    if any(t == 'features' for t in what):
        logs[3] = 1
    if any(t == 'results' for t in what):
        logs[4] = 1
    if any(t == 'tags' for t in what):
        logs[5] = 1
    return create_logfile(logs, path)


def blockPrint():
    """Suppress all print output after call."""
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    """Enable all print output after call, if it was blocked before."""
    sys.stdout = sys.__stdout__


def save_polydata_ply(what, where):
    """Save vtk.PolyData() as .ply file.

    File location can be specified.

    Parameters
    ----------
    what : vtk.PolyData()
        PolyData to be saved.

    where : string
        File location including file name without extension. .ply is automaticly
        added.
    """
    meshwriter = vtk.vtkPLYWriter()
    meshwriter.SetInput(what)
    meshwriter.SetFileName(where + '.ply')
    meshwriter.Write()

def readTiff(file_loc):
    return tiffread(file_loc)[0]

def curvature_max(mesh):
   curvature = vtk.vtkCurvatures()

   if vtk.VTK_MAJOR_VERSION <= 5:
     polyAlg = mesh.GetProducerPort()
     curvature.SetInputConnection(polyAlg)
   else:
     curvature.SetInputData(mesh)

   curvature.SetCurvatureTypeToMaximum()
   curvature.Update()
   vtkMaxVals = curvature.GetOutput().GetPointData().GetAbstractArray('Maximum_Curvature')
   maxVals = nps.vtk_to_numpy(vtkMaxVals)
   return maxVals

def curvature_min(mesh):
   curvature = vtk.vtkCurvatures()
   if vtk.VTK_MAJOR_VERSION <= 5:
     polyAlg = mesh.GetProducerPort()
     curvature.SetInputConnection(polyAlg)
   else:
     curvature.SetInputData(mesh)

   curvature.SetCurvatureTypeToMinimum()
   curvature.Update()
   vtkMinVals = curvature.GetOutput().GetPointData().GetAbstractArray('Minimum_Curvature')
   minVals = nps.vtk_to_numpy(vtkMinVals)
   return minVals

def curvature_gauss(mesh):
   curvature = vtk.vtkCurvatures()
   if vtk.VTK_MAJOR_VERSION <= 5:
     polyAlg = mesh.GetProducerPort()
     curvature.SetInputConnection(polyAlg)
   else:
     curvature.SetInputData(mesh)

   curvature.SetCurvatureTypeToGaussian()
   curvature.Update()
   vtkGaussVals = curvature.GetOutput().GetPointData().GetAbstractArray('Gauss_Curvature')
   gaussVals = nps.vtk_to_numpy(vtkGaussVals)
   return gaussVals

def curvature_mean(mesh):
   curvature = vtk.vtkCurvatures()

   if vtk.VTK_MAJOR_VERSION <= 5:
     polyAlg = mesh.GetProducerPort()
     curvature.SetInputConnection(polyAlg)
   else:
     curvature.SetInputData(mesh)
   curvature.SetCurvatureTypeToMean()
   curvature.Update()
   vtkMeanVals = curvature.GetOutput().GetPointData().GetAbstractArray('Mean_Curvature')
   meanVals = nps.vtk_to_numpy(vtkMeanVals)
   return meanVals

def get_connected_vertices(mesh, seed, includeSelf = True):
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

def get_curvatures(A, dist_ = 1, curv_types=['mean'], operations = [], ignore_boundary=False):

#  curv_types=['mean']
#  operations = []
#  ignore_boundary=True
#  #  A.mesh = calculate_curvature(A, curvature_type=curvature_type)
  A.calculate_curvatures(curv_types=curv_types, operations=operations)
  curvs = A.mesh.GetPointData().GetArray(A.curvature_type)
  curvs = pd.DataFrame(nps.vtk_to_numpy(curvs))

  ''' Compute graph '''
  G = nx.Graph()
  edges = []
  nPoints = A.mesh.GetNumberOfPoints()
  neighs = np.array([get_connected_vertices(A.mesh, ii) for ii in xrange(nPoints)])

  if ignore_boundary:
    boundary = boa.get_boundary_points(A.mesh)
    if len(boundary) > 0:
      curvs.iloc[boundary, 0] = float('Nan')

  # Make everything connect with its neighbours
  for key, value in enumerate(neighs):
    edges.extend([(key, ii) for ii in value if ii > key])
  G.add_edges_from(edges)

  ''' Average over extended neighborhood '''
  curvs = [curvs.iloc[np.array(nx.single_source_shortest_path_length(G, ii, cutoff=dist_).keys())].mean()[0] for ii in xrange(nPoints)]

  return pd.DataFrame(curvs)

'''
Example
=======
'''
#if __name__ == "__main__":
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
