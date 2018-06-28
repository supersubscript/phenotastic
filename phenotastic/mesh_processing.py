#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 22:10:18 2018

@author: henrik
"""
import numpy as np
from vtk.util import numpy_support as nps
import vtkInterface as vi
import Meristem_Phenotyper_3D as ap
from PyACVD import Clustering
import vtk
from scipy.ndimage.morphology import binary_fill_holes

# Preprocessing
def fill_contour(contour):
    ''' Assumes first dimension being z, ordered bottom to top. '''
    new_contour = contour.copy()

    # Close all sides but top
    new_contour[0] = 1
    #new_contour[-1] = 1
    new_contour[:, 0] = 1
    new_contour[:, -1] = 1
    new_contour[:, :, 0] = 1
    new_contour[:, :, -1] = 1

    for jj in xrange(new_contour.shape[1]):
        new_contour[:, jj] = binary_fill_holes(new_contour[:, jj])
    for jj in xrange(new_contour.shape[2]):
        new_contour[:, :, jj] = binary_fill_holes(new_contour[:, :, jj])

    new_contour[:, 0] = 0
    new_contour[:, -1] = 0
    new_contour[0] = 0
    new_contour[-1] = 0
    new_contour[:, :, 0] = 0
    new_contour[:, :, -1] = 0
    new_contour = binary_fill_holes(new_contour)
    return new_contour

# Actual mesh processing
def correct_bad_mesh(mesh):
    ''' Assumes a triangulated mesh. Note that recalculation of cell and point
    properties will have to be done '''
    from pymeshfix import _meshfix

    mesh.Clean()
    mesh.ExtractLargest()
    mesh.TriFilter()
    new_poly = mesh
    nm = get_non_manifold_edges(mesh)

    while nm.points.shape[0] > 0:
        print 'Trying to remove %d points' % nm.GetNumberOfPoints()
        meshfix = _meshfix.PyTMesh()

        comp_faces = np.delete(mesh.faces, np.arange(0, mesh.faces.size, 4))
        v, f = mesh.points, np.reshape(comp_faces, (comp_faces.shape[0] / 3, 3))

        meshfix.LoadArray(v, f)
        meshfix.RemoveSmallestComponents()
        v2, f2 = meshfix.ReturnArrays()
        f2 = np.hstack([np.append(len(ii), ii) for ii in f2])

        new_poly = vi.PolyData(v2, f2)
        new_poly.Clean()
        new_poly.ExtractLargest()

        # Try with forced point-removal
        nm = get_non_manifold_edges(new_poly)

        nmpts = nm.points
        mpts = new_poly.points
        ptidx = np.array([np.where((mpts == ii).all(axis=1))[0][0]
                          for ii in nmpts])
        mask = np.zeros((mpts.shape[0],), dtype=bool)
        if ptidx.shape[0] > 0:
            mask[ptidx] = True
        new_poly, _ = new_poly.RemovePoints(mask)

        new_poly.Clean()
        new_poly.ExtractLargest()
        new_poly.TriFilter()

        nm = get_non_manifold_edges(new_poly)

    new_poly.Clean()
    new_poly.ExtractLargest()
    new_poly.TriFilter()

    return new_poly

def remove_bridges(A, distance=1, threshold=1.0):
    ''' Assumes triangulated '''
    from domain_processing import get_boundary_points
    new_mesh = A.mesh

    while True:
        faces = new_mesh.faces.reshape(-1, 4).T[1:].T
        f_flat = faces.reshape(1,-1)
        boundary = get_boundary_points(new_mesh)
        border_faces = faces[np.unique(np.where(np.in1d(f_flat, boundary))[0] // 3)]

        # Find pts to remove
        all_boundary = np.array([np.all(np.in1d(ii, boundary)) for ii in border_faces])
        remove_pts = np.unique(border_faces[all_boundary].flatten())

        print('Removing %d points' % len(remove_pts))
        if len(remove_pts) == 0:
            break

        # Actually remove
        mask = np.zeros((new_mesh.GetNumberOfPoints(),), dtype=np.bool)
        mask[remove_pts] = True

        new_mesh = new_mesh.RemovePoints(mask, keepscalars=False)[0]
        new_mesh.ExtractLargest()
        new_mesh.Clean()
        new_mesh.Modified()
    return new_mesh

def remove_normals(A, threshold_angle=0, flip=False):

    A.compute_normals()
    negn = nps.vtk_to_numpy(A.mesh.GetPointData().GetNormals())
    if flip:
        negn *= -1
    negn = ap.cart2sphere(negn) / (2 * np.pi) * 360
    negn[:, 0] = 1

    to_remove = negn[:, 1] < threshold_angle
    A.mesh.RemovePoints(to_remove, keepscalars=False)[0]
    return A.mesh.RemovePoints(to_remove, keepscalars=False)[0]

def geodesic(mesh, ii, jj):
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(mesh)
    dijkstra.SetStartVertex(ii)
    dijkstra.SetEndVertex(jj)
    dijkstra.Update()

    arc = dijkstra.GetOutput()
    arc = vi.PolyData(arc)
    return arc

def remove_tongues_naive(A, radius, threshold, holefill=100):
    from scipy.spatial import KDTree
    mesh = A.mesh

    # Preprocessing
    mesh = remove_bridges(A)
    mesh.ExtractLargest()
    mesh.Clean()
    mesh.FillHoles(holefill)

    boundary = mesh.ExtractEdges(boundary_edges=True, non_manifold_edges=False,
                                 feature_edges=False, manifold_edges=False)

    # Get points and corresponding indices
    bdpts = boundary.points
    pts = A.mesh.points
    mesh_idxs = np.array([mesh.FindPoint(ii) for ii in bdpts])

    # Get the neighbours and then the number of adjacent boundary indices
    tree = KDTree(bdpts)
    neighs = tree.query_ball_point(pts, radius)
    nneighs = np.array([len([jj for jj in ii if jj in mesh_idxs]) for ii in neighs])

    to_remove = nneighs > threshold
    A.mesh = A.mesh.RemovePoints(to_remove, keepscalars=False)[0]

    # Postprocessing
    mesh = remove_bridges(A)
    mesh = correct_bad_mesh(A.mesh)
    mesh.ExtractLargest()
    mesh.Clean()
    mesh.FillHoles(holefill)
    return mesh

def remove_tongues(A, radius=30, threshold=6, threshold2=0.8, holefill=100):
    import networkx as nx
    from scipy.spatial import KDTree

    mesh = A.mesh

    while True:
        # Preprocessing
        mesh = remove_bridges(A)
        mesh.ExtractLargest()
        mesh.Clean()
        mesh.FillHoles(holefill)

        # Get boundary information and index correspondences
        boundary = mesh.ExtractEdges(boundary_edges=True, non_manifold_edges=False,
                                     feature_edges=False, manifold_edges=False)
        bdpts = boundary.points
        from_ = np.array([mesh.FindPoint(ii) for ii in bdpts])
        npts = boundary.points.shape[0]

        # Find the cycles, i.e. the different boundaries we have
        neighs = []
        ids = vtk.vtkIdList()
        for ii in xrange(npts):
            boundary.GetCellPoints(ii, ids)
            neighs.append([ids.GetId(0), ids.GetId(1)])

        net = nx.DiGraph(neighs)
        cycles = list(nx.simple_cycles(net))
        cycles.sort(key=lambda x: len(x), reverse=True)
        cycles = np.array([np.array(ii) for ii in cycles])

        # Loop over the cycles and find boundary points within radius
        to_remove = []
        for ii in xrange(len(cycles)):
            cpts = bdpts[cycles[ii]]

            # Get the boundary points (in same loop) within a certain radius
            tree = KDTree(cpts)
            neighs = tree.query_ball_point(cpts, radius)
            neighs = np.array([np.array(neighs[jj]) for jj in xrange(len(neighs))])
            neighs = np.array([neighs[jj][neighs[jj] != jj] for jj in xrange(len(neighs))])

            # Get and compare the euclidean and geodesic distance
            eucdists = np.array([np.sqrt(np.sum((cpts[jj] - cpts[neighs[jj]])**2, axis=1)) for jj in xrange(len(neighs))])

            geodists = []
            for jj in xrange(len(cpts)):
                geodists.append(np.array([geodesic(boundary, cycles[ii][jj], cycles[ii][neighs[jj][kk]]).GetLength() for kk in xrange(len(neighs[jj]))]))
            geodists = np.array(geodists)

            frac = np.array([geodists[jj] / eucdists[jj] for jj in xrange(len(neighs))])

            # Find which ones to (possibly remove)
            removal_anchors = []
            removal_geodists = []
            for kk in xrange(len(frac)):
                for jj in xrange(len(frac[kk])):
                    if frac[kk][jj] > threshold:
                        removal_anchors.append((kk, neighs[kk][jj]))
                        removal_geodists.append(geodists[kk][jj])
            removal_anchors = np.array(removal_anchors)
            removal_geodists = np.array(removal_geodists)

            for jj in xrange(len(removal_anchors)):
                gd = geodesic(mesh, from_[cycles[ii][removal_anchors[jj][0]]], from_[cycles[ii][removal_anchors[jj][1]]])
                shortest_geo = gd.GetLength()
                if shortest_geo / removal_geodists[jj] < threshold2:
                    gdpts = gd.points
                    to_remove.extend([mesh.FindPoint(kk) for kk in gdpts])
        to_remove = np.array(to_remove)

        if len(to_remove) == 0:
            break

        rmbool = np.zeros(mesh.points.shape[0], dtype=np.bool)
        rmbool[to_remove] = True

        A.mesh = mesh.RemovePoints(rmbool, keepscalars=False)[0]
        mesh = A.mesh
        mesh = remove_bridges(A)
        mesh = correct_bad_mesh(mesh)
        mesh.ExtractLargest()
        mesh.Clean()
        mesh.FillHoles(holefill)

    return mesh

def adjust_skirt(mesh, maxdist):
    boundary = mesh.ExtractEdges(non_manifold_edges=False, feature_edges=False,
                                 manifold_edges=False)
    lowest = mesh.GetBounds()[0]

    mpts = mesh.points
    bdpts = boundary.points
    idx_in_parent = np.array([mesh.FindPoint(ii) for ii in bdpts])

    to_adjust = idx_in_parent[bdpts[:, 0] - lowest < maxdist]
    mpts[to_adjust, 0] = lowest

    newmesh = vi.PolyData(mpts, mesh.faces)

    return newmesh


def remesh(mesh, npoints, subratio=10, max_iter=10000):
#    mesh = correct_bad_mesh(mesh)
    cobj = Clustering.Cluster(mesh)
    cobj.GenClusters(npoints, subratio=subratio, max_iter=max_iter)
    cobj.GenMesh()
    newmesh = vi.PolyData(cobj.ReturnMesh())
    newmesh.FillHoles(100)
    newmesh.ExtractLargest()
    newmesh.Clean()
    return newmesh


def remesh_decimate(mesh, iters):
    B = ap.AutoPhenotype()

    for ii in xrange(iters):
        B.mesh = correct_bad_mesh(mesh)
        B.mesh = remesh(B.mesh, B.mesh.points.shape[0] * 2)
        B.compute_normals()
        B.quadric_decimation(dec=.5, method="percentage")
        B.mesh.ExtractLargest()
        B.mesh.Clean()
        B.compute_normals()
        mesh = B.mesh

    return B.mesh


def get_non_manifold_edges(mesh):
    edges = mesh.ExtractEdges(boundary_edges=False,
                           non_manifold_edges=True, feature_edges=False,
                           manifold_edges=False)
    return edges

def get_boundary_edges(mesh):
    edges = mesh.ExtractEdges(boundary_edges=True,
                           non_manifold_edges=False, feature_edges=False,
                           manifold_edges=False)
    return edges

def get_manifold_edges(mesh):
    edges = mesh.ExtractEdges(boundary_edges=False,
                              non_manifold_edges=False, feature_edges=False,
                              manifold_edges=True)
    return edges

def get_feature_edges(mesh, angle=30):
    edges = mesh.ExtractEdges(feature_angle=angle, boundary_edges=False,
                              non_manifold_edges=False, feature_edges=True,
                              manifold_edges=False)
    return edges

def ECFT(mesh, holesize=100.0, inplace=False):
    if not inplace:
        mesh.Copy(mesh)

    mesh.ExtractLargest()
    mesh.Clean()
    mesh.FillHoles(holesize)
    mesh.TriFilter()

    if not inplace:
        return mesh

def define_meristem(mesh, pointData, method='central_mass', res=(0,0,0), fluo=None):
    ccoord = np.zeros((3,))
    if method == 'central_mass':
        com = vtk.vtkCenterOfMass()
        com.SetInputData(mesh)
        com.Update()
        ccoord = np.array(com.GetCenter())
    elif method == "central_space":
        ccoord = np.multiply(np.array(fluo.shape), np.array(res)) / 2
    elif method == 'central_bounds':
        ccoord = np.mean(np.reshape(mesh.GetBounds(), (3, 2)), axis=1)

    meristem = np.argmin(np.sqrt(np.sum((pointData[['z', 'y', 'x']] -
                                         ccoord)**2, axis=1)))
    meristem = pointData.loc[meristem, 'domain']
    return meristem, ccoord

def paraboloid_fit_mersitem(mesh):
    popt = ap.fit_paraboloid(mesh.points, )
    apex = ap.get_paraboloid_apex(popt)
    return popt, apex