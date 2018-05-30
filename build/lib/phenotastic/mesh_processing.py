#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 22:10:18 2018

@author: henrik
"""
import numpy as np
import vtkInterface as vi
import Meristem_Phenotyper_3D as ap
from PyACVD import Clustering
import vtk

def correct_bad_mesh(mesh):
    ''' Assumes a triangulated mesh. Note that recalculation of cell and point
    properties will have to be done '''
    from pymeshfix import _meshfix

    mesh.Clean()
    mesh.ExtractLargest()
    mesh.TriFilter()
    new_poly = mesh
    nm = get_non_manifold(mesh)

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
        nm = get_non_manifold(new_poly)

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

        nm = get_non_manifold(new_poly)

    return new_poly

def remesh(mesh, npoints, subratio=10, max_iter=10000):
    mesh = correct_bad_mesh(mesh)
    cobj = Clustering.Cluster(mesh)
    cobj.GenClusters(npoints, subratio=subratio, max_iter=max_iter)
    cobj.GenMesh()
    return vi.PolyData(cobj.ReturnMesh())


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


def get_non_manifold(mesh):
    fe = mesh.ExtractEdges(feature_angle=0, boundary_edges=False,
                           non_manifold_edges=True, feature_edges=False,
                           manifold_edges=False)
    return fe

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