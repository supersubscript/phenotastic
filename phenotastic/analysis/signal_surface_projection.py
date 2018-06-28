#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on 4 Aug 2017

@author: henrikahl
'''
import os
import numpy as np
import vtk
from phenotastic import Meristem_Phenotyper_3D as ap
#from phenotastic import Meristem_Phenotyper_3D as ap
import pandas as pd
import copy
#import handy_functions as hf
from skimage import measure
from vtk.util import numpy_support as nps
import tifffile as tiff
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, inverse_gaussian_gradient, checkerboard_level_set
#from scipy.ndimage.morphology import binary_fill_holes
import vtkInterface as vi
from skimage.exposure import equalize_hist
import phenotastic.domain_processing as boa
import phenotastic.plot as pl
import phenotastic.mesh_processing as mp
import phenotastic.misc

''' FILE INPUT '''
home = os.path.expanduser('~')
dir_ = os.path.abspath(os.path.join(home, 'projects', 'Meristem_Phenotyper_3D'))
file_ = os.path.abspath(os.path.join(dir_, 'test_images', 'meristem_test.tif'))

#dir_ = '/home/henrik/filipa/channel1/'
#file_ = dir_ + 'C1_Col0-1-1-Soil-3-3-Light-11_topcrop.tif'

file_ = "/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/C3_pWUS-3XVENUS-pCLV3-mCherry-off_NPA-3-18h-2.0-750-0.5-770_topcrop_deconv.tif"

#file_ = '/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/C1_pWUS-3XVENUS-pCLV3-mCherry-off_NPA-1-18h-2.0-750-0.5-770_topcrop_deconv.tif'
#data = tiff.imread(file_)
#wus = data[:, 0]
wfile_ = "/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/C1_pWUS-3XVENUS-pCLV3-mCherry-off_NPA-3-18h-2.0-750-0.5-770_topcrop_deconv.tif"
cfile_ = "/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/C2_pWUS-3XVENUS-pCLV3-mCherry-off_NPA-3-18h-2.0-750-0.5-770_topcrop_deconv.tif"

#file_ = '/home/henrik/data/180312-pWUS-3XVENUS-pCLV3-mCherry-Timelapse-6h_deconvolved_tiffs/C3_pWUS-3XVENUS-pCLV3-mCherry-off_NPA-1-18h-2.0-750-0.5-770_topcrop_deconv.tif'

''' Import data '''
inFile = tiff.TiffFile(file_)
data = inFile.asarray()
meta = inFile.imagej_metadata
data = data.astype(np.uint16)
#data[data < 3*np.mean(data)] = 0

if data.ndim == 4:
    fluo = data[:, 0]
elif data.ndim == 5:
    fluo = data[:, 0, 0]
else:
    fluo = data

winFile = tiff.TiffFile(wfile_)
wdata = winFile.asarray()
wmeta = winFile.imagej_metadata
wdata = wdata.astype(np.uint16)

if wdata.ndim == 4:
    wus = wdata[:, 0]
elif wdata.ndim == 5:
    wus = wdata[:, 0, 0]
else:
    wus = wdata
wus = wus.astype(np.float)

#cinFile = tiff.TiffFile(cfile_)
#cdata = cinFile.asarray()
#cmeta = cinFile.imagej_metadata
#cdata = cdata.astype(np.uint16)
#
#if cdata.ndim == 4:
#    clv = cdata[:, 0]
#elif cdata.ndim == 5:
#    clv = cdata[:, 0, 0]
#else:
#    clv = cdata
#clv = clv.astype(np.float)

fluo = np.maximum(wus, fluo)
#fluo = np.maximum(clv, fluo)



#tiff.imshow(equalize_hist(data))


''' Create AutoPhenotype object to store the data in '''
A = ap.AutoPhenotype()
A.data = fluo.copy()
A.data = A.data.astype(np.float)

''' Process data before creating contour. '''
A.data = equalize_hist(A.data, mask=A.data > np.mean(A.data))
for ii in xrange(3):
    A.data = median_filter(A.data, size=1)

for ii in xrange(3):
    A.data = gaussian_filter(A.data, sigma=[1, 1, 1])

''' Smooth the data (to fill holes) and create a contour. lambda2 > lambda1:
  more variable on inside. Smoothing might have to be corrected for different
  xy-z dimensionality. Iterations should ideally be at least 10, smoothing
  around 4. '''
A.data = (A.data - np.min(A.data)) / (np.max(A.data) - np.min(A.data))
A.data = A.data * np.max(fluo)

factor = 3
contour = morphological_chan_vese(A.data, iterations=1,
                                  init_level_set=A.data > factor *
                                  np.mean(A.data),
                                  smoothing=2, lambda1=10, lambda2=100)

#from scipy.ndimage.morphology import binary_fill_holes

for ii in xrange(contour.shape[0]):
    contour[ii] = binary_fill_holes(contour[ii])
contour = mp.fill_contour(contour)

#tiff.imshow(contour)

# tiff.imshow(A.data)

# tiff.imshow(contour)

A.contour = contour.copy()
A.contour = A.contour.astype(np.float)

''' Generate mesh '''
from stackAlign import imageProc as ip
#x,y,z = ip.resolution_xyz(file_)
#x, y, z = 1, 1, 2
x, y, z = 0.2516, 0.2516, 0.238
xyzres = z, y, x

''' Run MarchingCubes in skimage and convert to VTK format '''
verts, faces, normals, values = measure.marching_cubes_lewiner(
    A.contour, 0, spacing=([z, y, x]), step_size=1,
    allow_degenerate=False)
faces = np.hstack(np.c_[[len(ii) for ii in faces], faces])
surf = vi.PolyData(verts, faces)

''' Process mesh '''
A.mesh = surf
A.mesh.ExtractLargest()
bounds = A.mesh.GetBounds()

#first_center = contour[:, contour.shape[1] / 2, contour.shape[2] / 2]
#first_center = contour.shape[0] - first_center[::-1].argmax()
#A.mesh.ClipPlane([first_center * z - 40 * z, 0, 0], [xyzres[0], 0, 0])
# A.mesh.Plot(showedges=False)

bounds = A.mesh.GetBounds()
A.mesh.ClipPlane([np.ceil(bounds[0]), 0, 0], [xyzres[0], 0, 0])
A.mesh.ClipPlane([np.floor(bounds[1]), 0, 0], [-xyzres[0], 0, 0])
A.mesh.ClipPlane([0, np.ceil(bounds[2]), 0], [0, xyzres[1], 0])
A.mesh.ClipPlane([0, np.floor(bounds[3]), 0], [0, -xyzres[1], 0])
A.mesh.ClipPlane([0, 0, np.ceil(bounds[4])], [0, 0, xyzres[2]])
A.mesh.ClipPlane([0, 0, np.floor(bounds[5])], [0, 0, -xyzres[2]])
A.mesh.ExtractLargest()
A.mesh.FillHoles(50.0)
A.compute_normals()
# A.mesh.SetNormals(cell_normals=True, point_normals=True, split_vertices=False,
#                  flip_normals=True, consistent_normals=True,
#                  auto_orient_normals=True, non_manifold_traversal=True)
A.mesh.Clean()
A.mesh = A.mesh.Decimate(0.95, volume_preservation=True, normals=True)
A.mesh = mp.remesh(A.mesh, A.mesh.points.shape[0])
A.compute_normals()
A.mesh.Plot()

# A.mesh.SetNormals(cell_normals=True, point_normals=True, split_vertices=False,
#                  flip_normals=True, consistent_normals=True,
#                  auto_orient_normals=True, non_manifold_traversal=True)

A.mesh = mp.correct_bad_mesh(A.mesh)
A.mesh.FillHoles(10.0)
A.compute_normals()
A.smooth_mesh(iterations=1000, relaxation_factor=.01, boundarySmoothing=False,
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

#A.mesh.Plot(scalars=curvs)

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

pdata = boa.merge_boas_depth(A, pdata, threshold=0.1)
boas, boasData = boa.get_boas(pdata)
print boa.nboas(pdata)

pdata = boa.merge_boas_distance(pdata, boas, boasData, 15)
boas, boasData = boa.get_boas(pdata)
print boa.nboas(pdata)

pdata = boa.merge_boas_engulfing(A, pdata, threshold=0.6)
boas, boasData = boa.get_boas(pdata)
print boa.nboas(pdata)

pdata = boa.remove_boas_size(pdata, .02, method="relative_largest")
boas, boasData = boa.get_boas(pdata)
print boa.nboas(pdata)

''' Visualise '''
print boa.nboas(pdata)
print boa.boas_npoints(pdata)
#A.show_curvatures(stdevs = "all", curvs = curvs)
boaCoords = np.array([tuple(ii) for ii in boasData[['z', 'y', 'x']].values])

#pl.PlotPointData(A.mesh, pdata, 'domain',
#                 boaCoords=boaCoords, show_boundaries=True)

################################################################################
################################################################################
################################################################################
''' Export segmentation data '''
# Extract meristem
meristem_index, _ = boa.define_meristem(
    A.mesh, pdata, method='central_mass', fluo=fluo)
mpoly = boa.get_domain(A.mesh, pdata, meristem_index)

# Find geometrical apex by fitting paraboloid
popt, apex = mp.paraboloid_fit_mersitem(mpoly)
center_coord = mpoly.points[np.argmin(
    np.sqrt(np.sum((mpoly.points - apex)**2, axis=1)))]

# Extract domain data
ddata = boa.extract_domaindata(pdata, A.mesh, apex, meristem_index)
res = np.array([360 - ii if np.abs(360 - ii - 137.5) < np.abs(ii - 137.5)
                else ii for ii in np.abs(np.diff(ddata.angle.values))])
#print(np.mean(res[~np.isnan(res)]))

# Merge based on domain angles
#pdata, ddata = boa.merge_boas_angle(pdata, ddata, A.mesh, 7.5, apex, meristem_index)

################################################################################
################################################################################
################################################################################
#wus[wus < np.mean(wus)*3] = 0
#mask = contour.copy()
#mask[wus < np.mean(wus)*3] = 0
#pl.PlotImage(wus, xyzres, mask=mask, opacity=.4, psize=.5, mesh=None, meshopacity=.9)

coords = pl.coord_array(wus, xyzres)
vals = wus.ravel().copy()
vals[vals < 3 * np.mean(vals)] = 0
#pobj = vi.PlotClass()
#pobj.AddMesh(A.mesh)
#pobj.AddPoints(coords, scalars=vals, psize=.1, opacity=.1)
#pobj.Plot()

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

filter_ = np.logical_and(dists > -6, dists < 0)

#idxs = idxs[~np.isinf(dists)]
coords = coords[filter_]
vals = vals[filter_]
dists = dists[filter_]
#coords[idxs]
pobj = vi.PlotClass()
pobj.AddMesh(A.mesh)
pobj.AddPoints(coords, scalars=vals, opacity=1)


tree = cKDTree(A.mesh.points)
closest = tree.query(coords, k=1)[1]

sumvals = np.zeros(A.mesh.points.shape[0])
for ii in xrange(len(coords)):
    sumvals[closest[ii]] += vals[ii]

A.mesh.Plot(scalars=sumvals, background='white')

###########################

para = vtk.vtkQuadric()
p1, p2, p3, p4, p5, alpha, beta, gamma = popt

quadric = vtk.vtkQuadric()
quadric.SetCoefficients(p1, p2, 0, 0, 0, 0, p3, p4, -1, p5)

sample = vtk.vtkSampleFunction()
sample.SetSampleDimensions([200,200,200])
sample.SetImplicitFunction(quadric)
sample.SetModelBounds([-200, 200]*3)
sample.Update()

contour = vtk.vtkContourFilter()
contour.SetInputData(sample.GetOutput())
contour.Update()

contourMapper = vtk.vtkPolyDataMapper()
contourMapper.SetInputData(contour.GetOutput())
contourActor = vtk.vtkActor()
contourActor.SetMapper(contourMapper)

rotMat = ap.rot_matrix_44([alpha, beta, gamma], invert=True)
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
#tpoly.ClipPlane([self.GetBounds()[0] - 20, 0,0], [1,0,0])


bounds = A.mesh.GetBounds()
tpoly.ClipPlane([bounds[0] - 3, 0, 0], [1,0,0])
#tpoly = mp.remesh(tpoly, tpoly.points.shape[0])

pobj = vi.PlotClass()
pobj.AddMesh(tpoly, color = 'red', opacity=.1)
pobj.AddMesh(A.mesh, color = 'blue', opacity=.9)
pobj.SetBackground('white')
pobj.Plot()






