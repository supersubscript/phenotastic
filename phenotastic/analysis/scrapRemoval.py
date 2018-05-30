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
#import handy_functions as hf
from skimage import measure
from vtk.util import numpy_support as nps
import tifffile as tiff
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, inverse_gaussian_gradient, checkerboard_level_set
from scipy.ndimage.morphology import binary_fill_holes
import vtkInterface as vi
from skimage.exposure import equalize_hist
import domain_processing as boa
import plot as pl
import mesh_processing as mp

''' FILE INPUT '''
home = os.path.expanduser('~')
dir_ = os.path.abspath(os.path.join(home, 'projects', 'Meristem_Phenotyper_3D'))
file_ = os.path.abspath(os.path.join(dir_, 'test_images', 'meristem_test.tif'))

dir_ = '/home/henrik/filipa/channel1/'
file_ = dir_ + 'C1_Col0-1-1-Soil-3-3-Light-11_topcrop.tif'
# file_ = os.path.abspath(os.path.join(home, 'data', 'plant2', "00hrs_plant2_trim-acylYFP_hmin_2_asf_1_s_1.50_clean_3.tif")) # NPA meristem
# file_ = os.path.abspath(os.path.join(home, 'data', "20171102-FM4-64-cmu1cmu2-2-15_sam.tif")) # CMU1CMU2 mutant
# file_ = os.path.abspath(os.path.join(home, 'data', "20171103-FM4-64-Col0-2-15_sam.tif")) # WT mutant
# file_ = os.path.abspath(os.path.join(home, 'data', "C2-WUS-GFP-24h-light-1-1-Soil-1.tif")) # One of Benoit's
# file_ = os.path.abspath(os.path.join(home, 'data', "C2-WUS-GFP-24h-light-2-3-Soil-1-3-Sand-9.tif")) # Another of Benoits. Worse quality.

#file_= '/home/henrik/projects/stackAlign/data/pWUS-3X-VENUS-pCLV3-mCherry-on-NPA-2-0h-mCherry-0.7-Gain-800_topcrop.tif'
#file_ = '/home/henrik/pWUS-3XVENUS-pCLV3-mCherry-off_NPA-5-18h-2.0-750-0.5-770.tif'
#data = tiff.imread(file_)
#wus = data[:, 0]
#clv3 = data[:, 1]
# fluo = data[:, 2] # Take out autofluorescence

#file_ = '/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/C1_pWUS-3XVENUS-pCLV3-mCherry-off_NPA-1-18h-2.0-750-0.5-770_topcrop_deconv.tif'
#data = tiff.imread(file_)
#wus = data[:, 0]

#file_ = '/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/C3_pWUS-3XVENUS-pCLV3-mCherry-off_NPA-1-18h-2.0-750-0.5-770_topcrop_deconv.tif'

''' Import data '''
inFile = tiff.TiffFile(file_)
data = inFile.asarray()
meta = inFile.imagej_metadata
data = data.astype(np.uint16)

if data.ndim == 4:
    fluo = data[:, 0]
elif data.ndim == 5:
    fluo = data[:, 0, 0]
else:
    fluo = data
fluo = fluo.astype(np.uint16)

''' Create AutoPhenotype object to store the data in '''
A = ap.AutoPhenotype()
A.data = fluo

''' Process data before creating contour. '''
A.data = equalize_hist(A.data)
for ii in xrange(1):
    A.data = gaussian_filter(A.data, sigma=[1.5, 3, 3])

''' Smooth the data (to fill holes) and create a contour. lambda2 > lambda1:
  more variable on inside. Smoothing might have to be corrected for different
  xy-z dimensionality. Iterations should ideally be at least 10, smoothing
  around 4. '''

A.data = A.data * np.max(fluo)
A.data = (A.data - np.min(A.data)) / (np.max(A.data) - np.min(A.data))

factor = 3.0
contour = morphological_chan_vese(A.data, iterations=1,
                                  init_level_set=A.data > factor *
                                  np.mean(A.data),
                                  smoothing=1, lambda1=10, lambda2=100)
# tiff.imshow(contour)
# tiff.imshow(A.data)

# Close all sides but top
contour[0] = 1
#contour[-1] = 1
contour[:, 0] = 1
contour[:, -1] = 1
contour[:, :, 0] = 1
contour[:, :, -1] = 1

for jj in xrange(contour.shape[1]):
    contour[:, jj] = binary_fill_holes(contour[:, jj])
for jj in xrange(contour.shape[2]):
    contour[:, :, jj] = binary_fill_holes(contour[:, :, jj])

contour[:, 0] = 0
contour[:, -1] = 0
contour[0] = 0
contour[-1] = 0
contour[:, :, 0] = 0
contour[:, :, -1] = 0
contour = binary_fill_holes(contour)

# tiff.imshow(contour)

A.contour = contour.copy()
A.contour = A.contour.astype(np.float)

''' Generate mesh '''
#x,y,z = ip.resolution_xyz(file_)
x, y, z = 1, 1, 2
xyzres = z, y, x

''' Run MarchingCubes in skimage and convert to VTK format '''
verts, faces, normals, values = measure.marching_cubes_lewiner(
    A.contour, 0, spacing=([z, y, x]), step_size=1,
    allow_degenerate=False)
faces = np.hstack(np.c_[[len(ii) for ii in faces], faces])
surf = vi.PolyData(verts, faces)

''' Process mesh '''
A.mesh = surf
bounds = A.mesh.GetBounds()

first_center = contour[:, contour.shape[1] / 2, contour.shape[2] / 2]
first_center = contour.shape[0] - first_center[::-1].argmax()
A.mesh.ClipPlane([first_center * z - 40 * z, 0, 0], [xyzres[0], 0, 0])
# A.mesh.Plot(showedges=False)

bounds = A.mesh.GetBounds()
A.mesh.ClipPlane([np.ceil(bounds[0]), 0, 0], [xyzres[0], 0, 0])
A.mesh.ClipPlane([np.floor(bounds[1]), 0, 0], [-xyzres[0], 0, 0])
A.mesh.ClipPlane([0, np.ceil(bounds[2]), 0], [0, xyzres[1], 0])
A.mesh.ClipPlane([0, np.floor(bounds[3]), 0], [0, -xyzres[1], 0])
A.mesh.ClipPlane([0, 0, np.ceil(bounds[4])], [0, 0, xyzres[2]])
A.mesh.ClipPlane([0, 0, np.floor(bounds[5])], [0, 0, -xyzres[2]])
A.mesh.ExtractLargest()
A.mesh.FillHoles(100.0)
A.compute_normals()
# A.mesh.SetNormals(cell_normals=True, point_normals=True, split_vertices=False,
#                  flip_normals=True, consistent_normals=True,
#                  auto_orient_normals=True, non_manifold_traversal=True)
A.mesh.Clean()
A.mesh = A.mesh.Decimate(0.95, volume_preservation=True, normals=True)
A.compute_normals()
# A.mesh.Plot()

# A.mesh.SetNormals(cell_normals=True, point_normals=True, split_vertices=False,
#                  flip_normals=True, consistent_normals=True,
#                  auto_orient_normals=True, non_manifold_traversal=True)

A.mesh = mp.correct_bad_mesh(A.mesh)
A.mesh.FillHoles(100)
A.compute_normals()
A.smooth_mesh(iterations=100, relaxation_factor=.01, boundarySmoothing=False,
              featureEdgeSmoothing=False, feature_angle=45)
A.mesh.Clean()
A.mesh.ExtractLargest()

# Sufficient loop to remesh
A.mesh = mp.remesh_decimate(A.mesh, iters=3)
A.mesh = mp.remesh(A.mesh, A.mesh.points.shape[0])
A.mesh.Clean()
A.mesh.ExtractLargest()
A.compute_normals()
################################################################################
################################################################################
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

A.mesh.Plot(scalars=curvs)

''' Create graphs '''
pointData = boa.init_pointdata(A, curvs, neighs)

''' Identify BoAs'''
pointData = boa.domains_from_curvature(pointData)
boas, boasData = boa.get_boas(pointData)
print boa.nboas(pointData)

''' Process boas '''
safeCopy = copy.deepcopy(pointData)
pointData = copy.deepcopy(safeCopy)
boas, boasData = boa.get_boas(pointData)

pointData = boa.merge_boas_depth(A, pointData, threshold=0.01)
boas, boasData = boa.get_boas(pointData)
print boa.nboas(pointData)

pointData = boa.merge_boas_distance(pointData, boas, boasData, 15)
boas, boasData = boa.get_boas(pointData)
print boa.nboas(pointData)

pointData = boa.merge_boas_engulfing(A, pointData, threshold=0.6)
boas, boasData = boa.get_boas(pointData)
print boa.nboas(pointData)

pointData = boa.remove_boas_size(pointData, .02, method="relative_largest")
boas, boasData = boa.get_boas(pointData)
print boa.nboas(pointData)

''' Visualise '''
print boa.nboas(pointData)
print boa.boas_npoints(pointData)
#A.show_curvatures(stdevs = "all", curvs = curvs)
boaCoords = np.array([tuple(ii) for ii in boasData[['z', 'y', 'x']].values])

pl.PlotPointData(A.mesh, pointData, 'domain',
                 boaCoords=boaCoords, show_boundaries=True)

################################################################################
################################################################################
################################################################################
''' Export segmentation data '''
meristem, _ = boa.define_meristem(
    A.mesh, pointData, method='central_mass', fluo=fluo)
mpoly = boa.get_domain(A.mesh, pointData, meristem)

popt, apex = boa.paraboloid_fit_mersitem(mpoly)
center_coord = mpoly.points[np.argmin(
    np.sqrt(np.sum((mpoly.points - apex)**2, axis=1)))]
domains = np.unique(pointData.domain)

domainData = pd.DataFrame(
    columns=['domain', 'dist_boundary', 'dist_com', 'angle', 'area', 'ismeristem'])
for ii in domains:
    # Get distance for closest boundary point to apex
    dom = boa.get_domain(A.mesh, pointData, ii)
    dom_boundary = boa.get_boundary_points(dom)
    dom_boundary_coords = dom.points[dom_boundary]
    dom_boundary_dists = np.sqrt(
        np.sum((dom_boundary_coords - apex)**2, axis=1))
    d2boundary = np.min(dom_boundary_dists)

    # Get distance for center of mass from apex
    center = dom.CenterOfMass()
    d2com = np.sqrt(np.sum((center - apex)**2))

    # Get domain angle in relation to apex
    rel_pos = center - apex
    angle = np.arctan2(rel_pos[1], rel_pos[2])  # angle in yz-plane
    if angle < 0:
        angle += 2. * np.pi
    angle *= 360 / (2. * np.pi)

    # Get surface area
    area = dom.SurfaceArea()

    # Define type
    ismeristem = ii == meristem
    if ismeristem:
        angle = 0

    # Set data
    domainData.loc[int(ii)] = [int(ii), d2boundary,
                               d2com, angle, area, ismeristem]

domainData = domainData.sort_values(['ismeristem', 'area'], ascending=False)
res = np.array([360 - ii if np.abs(360 - ii - 137.5) < np.abs(ii - 137.5)
                else ii for ii in np.abs(np.diff(domainData.angle.values))])  # golden ratio!

################################################################################
################################################################################
################################################################################
pl.PlotImage(fluo, xyzres, mask=contour, opacity=.1, psize=.5)

''' Plot image data '''
#
#''' When looking at intensity data, only look inside the contour. Here we create
#    a rectangle around this. We then input our intensity data to a VTK object.
#    '''
# Find bounds
#start = []
#stop = []
# for dd in xrange(contour.ndim):
#  rolled = np.rollaxis(contour, dd)
#  for ii in xrange(rolled.shape[0]):
#    if np.any(rolled[ii] > 0):
#      start.append(ii)
#      break
#  for ii in xrange(rolled.shape[0] - 1, -1, -1):
#    if np.any(rolled[ii] > 0):
#      stop.append(ii)
#      break

# Convert the VTK array to vtkImageData
#data = wus
#data_cropped = data[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]].ravel()
# vtk_data_array = nps.numpy_to_vtk(
#  num_array=data_cropped,
#  deep=True,
#  array_type=vtk.VTK_FLOAT)
#
#spacing = [0.22, 0.2516, 0.2516]
#img_vtk = vtk.vtkImageData()
#img_vtk.SetDimensions(np.subtract(stop, start))
#img_vtk.SetOrigin(np.multiply(start, spacing))
#img_vtk.SetSpacing([0.22, 0.2516, 0.2516])
# img_vtk.GetPointData().SetScalars(vtk_data_array)
#pts = np.array([img_vtk.GetPoint(ii) for ii in xrange(img_vtk.GetNumberOfPoints())])
#
#data_line = data_cropped
#pts = pts[data_line > 0]
#data_line = data_line[data_line > 0]
#
##del data, spacing, vtk_data_array, verts, values, rolled, normals, faces
#import gc; gc.collect()
#d = np.zeros((len(pts), 3), dtype='float64')
#dists = np.zeros(len(pts))
#outpt = np.zeros((3,), dtype='float64')
#
# for ii in xrange(len(pts)):
#  dists[ii] = implicit_function.EvaluateFunctionAndGetClosestPoint(pts[ii], outpt)
#  d[ii] = outpt
#
# A.fill_holes(1000)
# A.compute_normals()
#
#pointsPolydata = vtk.vtkPolyData()
#vpts = vtk.vtkPoints()
#vpts.SetData(nps.numpy_to_vtk(pts, deep=1, array_type=vtk.VTK_FLOAT))
# pointsPolydata.SetPoints(vpts)
#extPts = vtk.vtkSelectEnclosedPoints()
# extPts.SetInputData(pointsPolydata)
#
# extPts.SetSurfaceData(A.mesh)
# extPts.Update()
#inside = [extPts.IsInside(ii) for ii in xrange(len(pts))]
#inside = np.array(inside, dtype='bool')
#
#data_line = data_line[inside]
#pts = pts[inside]
#
#from scipy.spatial import cKDTree
#
#dists = np.zeros((len(pts), ))
#idxs = np.zeros((len(pts), ))
#
#tree = cKDTree(coords)
# for ii in xrange(len(pts)):
#  dists[ii], idxs[ii] = tree.query(pts[ii], k=1)
#
#filter_ = dists < 6
#dists = dists[filter_]
#idxs = idxs[filter_]
#data_line = data_line[filter_]
#
#vals = np.zeros((A.mesh.GetNumberOfPoints(), ))
# for ii in xrange(len(idxs)):
#  vals[idxs[ii].astype(np.int)] += data_line[ii]
#
#vtkVals = nps.numpy_to_vtk(vals, deep=True, array_type=vtk.VTK_FLOAT)
#
#''' TODO: Send into plot point vals '''
# neighs = np.array([ap.get_connected_vertices(A.mesh, ii) for ii in xrange(A.mesh.GetNumberOfPoints())]) # get linkage
#vals   = np.array([np.mean(vals[np.append(ii, neighs[ii])]) for ii in xrange(A.mesh.GetNumberOfPoints())])
#vals   = np.array([np.mean(vals[np.append(ii, neighs[ii])]) for ii in xrange(A.mesh.GetNumberOfPoints())])
#vals   = np.array([np.mean(vals[np.append(ii, neighs[ii])]) for ii in xrange(A.mesh.GetNumberOfPoints())])

#A.show_point_values(pd.DataFrame(vals), stdevs="all", logScale=True, ruler=True)


#########
#mapper = vtk.vtkDataSetMapper()
# mapper.SetInputData(img_vtk)
#actor = vtk.vtkActor()
# actor.SetMapper(mapper)
#
#renderWindow = vtk.vtkRenderWindow()
#
#renderer = vtk.vtkRenderer()
# renderWindow.AddRenderer(renderer)
# renderer.AddActor(actor)
# renderer.ResetCamera()
#renderWindowInteractor = vtk.vtkRenderWindowInteractor()
# renderWindowInteractor.SetRenderWindow(renderWindow)
# renderWindow.Render()
# renderWindowInteractor.Start()
#


# This is extremely slow v
#distances = np.zeros((len(pts), ))
#closestPts = np.zeros((len(pts),))
#import operator
# for ii in xrange(len(pts)):
#  distance = np.sqrt(np.sum((pts[ii] - coords)**2, axis=1))
#  min_index, min_value = min(enumerate(distance), key=operator.itemgetter(1))
#  distances[ii] = min_value
#  closestPts[ii] = min_index

#inside = [extPts.IsInsideSurface(ii) for ii in pts]


#d     = d[dists < 0]
#dists = dists[dists < 0]
#
#d     = d[dists > -6]
#dists = dists[dists > -6]

#assert(d.dtype == coords.dtype)
#
##matches = []
#count = 0
# for ii in d:
# for jj in coords:
# if np.all(ii == jj):
##      count += 1
# break
#l = []
# for ii in d:
#  l.append(np.where(np.all(ii == coords, axis=1)))

# def check_inside(implicit_function, x,y,z):
#  return implicit_function.FunctionValue([x,y,z]) <= 0


#
##A.smooth_mesh(iterations=300, relaxation_factor=.01)
# A.update_mesh()
# A.clean_mesh()
#
# A.show_mesh()
#dist_ = 1
#curvs = ap.get_curvatures(A, dist_=dist_, curv_types=['gauss', 'max'], operations=['-'], ignore_boundary=True)
#A.show_curvatures(stdevs="all", curvs=curvs)
#
#''' Create graphs '''
#nPoints = A.mesh.GetNumberOfPoints()
#coords = [A.mesh.GetPoint(ii) for ii in xrange(A.mesh.GetNumberOfPoints())]
#neighs = [ap.get_connected_vertices(A.mesh, ii) for ii in xrange(nPoints)]
#coords = pd.DataFrame(coords)
#neighs = pd.DataFrame(np.array(neighs))
#pointData = pd.concat([curvs, coords, neighs], axis=1)
#pointData.columns = ['curv', 'x', 'y', 'z', 'neighs']
#pointData['domain'] = np.NaN * nPoints
#
#''' Identify BoAs'''
#pointData      = boa.domains_from_curvature(pointData)
#boas, boasData = boa.get_boas(pointData)
##
##''' Process boas '''
#safeCopy = copy.deepcopy(pointData)
#pointData = copy.deepcopy(safeCopy)
#boas, boasData = boa.get_boas(pointData)

#pointData = boa.merge_boas_depth(pointData, threshold=0.008)
#pointData = boa.merge_boas_depth(pointData, threshold=0.010)
#boas, boasData = boa.get_boas(pointData)
#
#pointData = boa.merge_boas_engulfing(A, pointData, threshold=0.6)
#boas, boasData = boa.get_boas(pointData)
#
#pointData = boa.remove_boas_size(pointData, .10, method = "relative_meristem")
#boas, boasData = boa.get_boas(pointData)
##
##''' Visualise in VTK '''
# print boa.get_nBoas(pointData)
# print boa.get_boas_nPoints(pointData)
###A.show_curvatures(stdevs = "all", curvs = curvs)
#boaCoords = [tuple(x) for x in boasData[['x', 'y', 'z']].values]
# A.show_point_values(vals = pd.DataFrame(np.array(pointData['domain'])),
#                    discrete = True, boaCoords=boaCoords)
