# -*- coding: utf-8 -*-
"""
Created on Tue May 29 22:10:18 2018

@author: henrik
"""
import sys, os
os.chdir('/home/henrik/projects/surface_extraction/code/phenotastic/phenotastic')

import numpy as np
import pyvista
from vtk.util import numpy_support as nps
#from pyacvd import clustering
import vtk
import surface
from scipy.ndimage.morphology import binary_fill_holes
#import mesh_processing as mp

# Preprocessing
def fill_contour(contour, fill_xy=False, fill_zx_zy=False, inplace=False):
    """
    Fill contour by closing all the edges (except for the top one), and applying
    a binary fill-holes operation. Note that this causes some errors if there is
    significant curvature on the contour, since the above-signal is
    down-projected. This can cause some erronous sharp edges which ruin the
    contour.

    Parameters
    ----------
    contour : np.ndarray
        Contour to operate on.

    fill_xy : bool, optional
        Flag to also fill in the xy-plane. Note that this can fill actual holes
        that arise if for example two relatively distant primordia touch each
        other at a point that isn't close to the meristem.

    inplace : bool, optional
        Flag to modify object in place.

    Returns
    -------
    new_contour : np.ndarray
        Contour after modification. If inplace == True, nothing is returned.

    Notes
    -----
    Assumes first dimension being Z, ordered from bottom to top. Will remove top
    and bottom slice.


    """
    if not inplace:
        new_contour = contour.copy()
    else:
        new_contour = contour


    new_contour = np.pad(new_contour, 1, 'constant', constant_values=1)

    # Close all sides but top
#    new_contour[0] = 1
    new_contour[-1] = 0 # top
#    new_contour[:, 0] = 1
#    new_contour[:, -1] = 1
#    new_contour[:, :, 0] = 1
#    new_contour[:, :, -1] = 1

    # Fill holes form in xz & yz planes.
    if fill_zx_zy:
        for ii in range(new_contour.shape[1]):
            new_contour[:, ii] = binary_fill_holes(new_contour[:, ii])
        for ii in range(new_contour.shape[2]):
            new_contour[:, :, ii] = binary_fill_holes(new_contour[:, :, ii])

    # Remove edges again, also for top
    new_contour[0] = 0
    new_contour[-1] = 0
    new_contour[:, 0] = 0
    new_contour[:, -1] = 0
    new_contour[:, :, 0] = 0
    new_contour[:, :, -1] = 0

    if fill_xy:
        for ii in range(new_contour.shape[0]):
            new_contour[ii] = binary_fill_holes(new_contour[ii])

    new_contour = binary_fill_holes(new_contour)
    new_contour = new_contour[1:-1, 1:-1, 1:-1]

    if inplace:
        return
    else:
        return new_contour

### Actual mesh processing
def correct_bad_mesh(mesh, verbose=True):
    """
    Correct a bad (non-manifold) mesh with two methods:
        1) method removesmallcomponents from the pymeshfixpackage, and
        2) identifying leftover non-manifold edges and removing all the points
           in these.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh.

    verbose : bool, optional
        Flag to print out operation procedure.

    Notes
    -----
    - Assumes a triangulated mesh.
    - Recalculation of cell and point attributes will have to be redone
    - All points in non-manifold edges will be removed. This could in principle
      be improved upon, since one point may be sufficient to create a manifold
      mesh.

    Returns
    -------

    """
    try:
        from pymeshfix import _meshfix
    except ImportError:
        raise ImportError(
                'Package pymeshfix not found. Install to use this function.')

    new_poly = ECFT(mesh, 0)
    nm = get_non_manifold_edges(new_poly)

    while nm.n_points > 0:
        if verbose:
            print(('Trying to remove %d points' % nm.GetNumberOfPoints()))

        # Create pymeshfix object from our mesh
        meshfix = _meshfix.PyTMesh()
        v, f = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        meshfix.load_array(v, f)

        # Remove smaller components
        meshfix.remove_smallest_components()
        v2, f2 = meshfix.return_arrays()
        f2 = np.hstack([np.append(len(ii), ii) for ii in f2])

        # Create new polydata from cleaned out mesh
        new_poly = pyvista.PolyData(v2, f2)
        new_poly = ECFT(new_poly, 0)

        # If we still have non-manifold edges, force remove these points
        nm = get_non_manifold_edges(new_poly)
        nmpts = nm.points
        mpts = new_poly.points
        ptidx = np.array([np.where((mpts == ii).all(axis=1))[0][0]
                          for ii in nmpts])

        mask = np.zeros((mpts.shape[0],), dtype=bool)
        if ptidx.shape[0] > 0:
            mask[ptidx] = True
        new_poly = new_poly.remove_points(mask)[0]

        new_poly = ECFT(new_poly, 0)
        nm = get_non_manifold_edges(new_poly)

    new_poly = ECFT(new_poly, 0)

    return new_poly

# TODO: Add inplace argument
def remove_bridges(mesh, verbose=True):
    """
    Remove triangles where all vertices are part of the mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to operate on.

    verbose : bool, optional
        Flag to print processing steps.

    Notes
    -----
    Assumes triangulated mesh.

    Returns
    -------
    new_mesh : pyvista.PolyData
        Mesh after bridge removal.

    """
    new_mesh = mesh

    while True:
        # Retrieve triangles on the border
        faces = new_mesh.faces.reshape(-1, 4)[:, 1:]
        f_flat = faces.ravel()
        boundary = get_boundary_points(new_mesh)
        border_faces = faces[np.unique(np.where(np.in1d(f_flat, boundary))[0] // 3)]

        # Find pts to remove
        all_boundary = np.array([np.all(np.in1d(ii, boundary)) for ii in border_faces])
        remove_pts = np.unique(border_faces[all_boundary].flatten())

        if verbose:
            print(('Removing %d points' % len(remove_pts)))
        if len(remove_pts) == 0:
            break

        # Actually remove
        mask = np.zeros((new_mesh.n_points,), dtype=np.bool)
        mask[remove_pts] = True

        new_mesh = new_mesh.remove_points(mask, keep_scalars=False)[0]
        new_mesh = ECFT(new_mesh, 0)

    return new_mesh

def remove_normals(mesh, threshold_angle=0, flip=False, angle='polar'):
    """ Remove points based on the point normal angle.

    Currently only considering the polar angle.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh on which to operate.

    threshold_angle : float, optional
        Threshold for the polar angle (theta). Values smaller than this will be
        removed. Default = 0.

    flip : bool, optional
         Flag to flip normal orientation. Default = False.

    Returns
    -------
    new_mesh : pyvista.PolyData
        Mesh with the resulting vertices removed.

    """
    from misc import car2sph

    normals = mesh.point_normals.copy()
    if flip:
        normals *= -1.
    normals = car2sph(normals) / (2. * np.pi) * 360.

    if angle == 'polar':
        angle_index = 1
    elif angle == 'azimuth':
        angle_index = 2
    else:
        raise ValueError('Parameter \'angle\' can only take attributes \'polar\' and \'azimuth\'.')

    to_remove = normals[:, angle_index] < threshold_angle
    new_mesh = mesh.remove_points(to_remove, keep_scalars=False)[0]
    return new_mesh

#def smooth_boundary(mesh, lambda):
#    boundary = mp.get_boundary_edges(mesh)
#    bdpts = boundary.points
#    from_ = np.array([mesh.FindPoint(ii) for ii in bdpts])
#    npts = boundary.n_points
#
#    # Find the cycles, i.e. the different boundaries we have
#    neighs = []
#    ids = vtk.vtkIdList()
#    for ii in range(npts):
#        boundary.GetCellPoints(ii, ids)
#        neighs.append([ids.GetId(0), ids.GetId(1)])
#
#    net = nx.DiGraph(neighs)
#    cycles = list(nx.simple_cycles(net))
#    cycles.sort(key=lambda x: len(x), reverse=True)
#    cycles = np.array([np.array(ii) for ii in cycles])
def smooth_boundary(mesh, iterations=20, sigma=.1, inplace=False):
    """
    """
#    import pyvista as pv
#    mesh = pv.read('breaking_boundary_smooth.ply')
#    iterations=20
#    sigma=.1
#    inplace=False
    import networkx as nx

    mesh = mesh.copy() if not inplace else mesh

    # Get boundary information and index correspondences
    boundary = get_boundary_edges(mesh)
    bdpts = boundary.points
    from_ = np.array([mesh.FindPoint(ii) for ii in bdpts])
    npts = boundary.n_points

#     Find the cycles, i.e. the different boundaries we have
    neighs = [] #list(get_connected_vertices_all(boundary, include_self=False))
    for ii in range(boundary.n_points):
        pt_neighs = get_connected_vertices(boundary, ii, include_self=False)
        for jj in range(pt_neighs.shape[0]):
            neighs.append((ii, pt_neighs[jj]))

    net = nx.Graph(neighs)
    cycles = nx.cycle_basis(net)
    cycles.sort(key=lambda x: len(x), reverse=True)
    cycles = np.array([np.array(ii) for ii in cycles])

    new_pts_prev = bdpts.copy()
    new_pts_now = bdpts.copy()
    for iter in range(iterations):
        new_pts_prev = new_pts_now.copy()
        for ii in range(len(cycles)):
            for jj in range(len(cycles[ii])):
                new_pts_now[cycles[ii][jj]] = new_pts_prev[cycles[ii][jj]] - sigma * (new_pts_prev[cycles[ii][jj]] - np.mean(np.array([new_pts_prev[cycles[ii][jj]], new_pts_prev[cycles[ii][jj-1]], new_pts_prev[cycles[ii][(jj+1) % len(cycles[ii])]]]), axis=0))

    # update coordinates
    for ii in range(len(cycles)):
        mesh.points[from_[cycles[ii]]] = new_pts_now[cycles[ii]]

    return None if inplace else mesh


# TODO: Add inplace argument
def remove_tongues(mesh, radius, threshold=6,
                   verbose=True):

    """
    Remove "tongues" in mesh.

    All boundary points within a given radius are considered. The ones where the
    fraction of the distance along the boundary, as divided by the euclidean
    distance, is greater than the given threshold.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to operate on.

    radius : float
        Radius for boundary point neighbourhood.

    threshold : float, optional
        Threshold for fraction between boundary distance and euclidean distance.
        Default = 6.

    Returns
    -------
    mesh : pyvista.PolyData
        Resulting mesh.

    """
    import networkx as nx
    from scipy.spatial import KDTree

    while True:
        # Get boundary information and index correspondences
        boundary = get_boundary_edges(mesh)
        bdpts = boundary.points
        from_ = np.array([mesh.FindPoint(ii) for ii in bdpts])

        # Find the cycles, i.e. the different boundaries we have
#        neighs = []
        neighs = [] #list(get_connected_vertices_all(boundary, include_self=False))
        for ii in range(boundary.n_points):
            pt_neighs = get_connected_vertices(boundary, ii, include_self=False)
            for jj in range(pt_neighs.shape[0]):
                neighs.append((ii, pt_neighs[jj]))

        net = nx.Graph(neighs)
        cycles = nx.cycle_basis(net)
        cycles.sort(key=lambda x: len(x), reverse=True)
        cycles = np.array([np.array(ii) for ii in cycles])
#        boundary.plot(notebook=False, scalars=np.isin(np.arange(boundary.n_points), cycles[-1]).astype('int'), line_width=3)

        # Loop over the cycles and find boundary points within radius
        to_remove = []
        for ii in range(len(cycles)):
            print('Running cycle {} with {} points'.format(ii, len(cycles[ii])))
            cpts = bdpts[cycles[ii]]

            # Get the boundary points (in same loop) within a certain radius
            tree = KDTree(cpts)
            neighs = tree.query_ball_point(cpts, radius)
            neighs = np.array([np.array(neighs[jj]) for jj in range(len(neighs))])
            neighs = np.array([neighs[jj][neighs[jj] != jj] for jj in range(len(neighs))])

            # Get and compare the euclidean and geodesic distance
            eucdists = np.array([np.sqrt(np.sum((cpts[jj] - cpts[neighs[jj]])**2, axis=1)) for jj in range(len(neighs))])

            geodists = []
            for jj in range(len(cpts)):
                geodists.append(np.array([
                        boundary.geodesic_distance(cycles[ii][jj],
                                                   cycles[ii][neighs[jj][kk]])
                        for kk in range(len(neighs[jj]))]))
            geodists = np.array(geodists)

            frac = np.array([geodists[jj] / eucdists[jj] for jj in range(len(neighs))])

            # Find which ones to (possibly remove)
            removal_anchors = []
            removal_geodists = []
            for kk in range(len(frac)):
                for jj in range(len(frac[kk])):
                    if frac[kk][jj] > threshold:
                        removal_anchors.append((kk, neighs[kk][jj]))
                        removal_geodists.append(geodists[kk][jj])
            removal_anchors = np.array(removal_anchors)
            removal_geodists = np.array(removal_geodists)

            for jj in range(len(removal_anchors)):
#                gd = boundary.geodesic(from_[cycles[ii][removal_anchors[jj][0]]], from_[cycles[ii][removal_anchors[jj][1]]])
                gd = boundary.geodesic(cycles[ii][removal_anchors[jj][0]], cycles[ii][removal_anchors[jj][1]])
#                shortest_geo = gd.GetLength()
#                if shortest_geo / removal_geodists[jj] < threshold2:
                gdpts = gd.points
                to_remove.extend([mesh.FindPoint(kk) for kk in gdpts])
        to_remove = np.unique(to_remove)

        if len(to_remove) == 0:
            break

        # Remove points
        mesh = mesh.remove_points(to_remove, keep_scalars=False)[0]
#        mesh = remove_bridges(mesh)
        mesh = ECFT(mesh, 0)
        mesh = correct_bad_mesh(mesh)
        mesh = ECFT(mesh, 0)

    return mesh

def repair(mesh):
    import pymeshfix as pmf
    tmp = pmf.MeshFix(mesh)
    tmp.repair(True)
    return tmp.mesh

def remesh(mesh, n, sub=3):
    from pyacvd import clustering
    clus = clustering.Clustering(mesh)
    clus.subdivide(sub)  # 2 also works
    clus.cluster(n)
    output = clus.create_mesh()
    return output

def make_manifold(mesh, hole_size=100):
    mesh = mesh.copy()
    edges = mesh.extract_edges(boundary_edges=False,
                               feature_edges=False,
                               manifold_edges=False,
                               non_manifold_edges=True)
    while edges.n_points > 0:
        to_remove = [mesh.FindPoint(pt) for pt in edges.points]
        print('Removing {} points'.format(len(to_remove)))
        mesh = mesh.remove_points(to_remove)[0]
        mesh = mesh.extract_largest()
        mesh = mesh.fill_holes(hole_size)
        mesh = mesh.clean()
        edges = mesh.extract_edges(boundary_edges=False,
                                   feature_edges=False,
                                   manifold_edges=False,
                                   non_manifold_edges=True)

    return mesh



def drop_skirt(mesh, maxdist, flip=False):
    """
    Downprojects the boundary to the lowest point in the z-direction.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to operate on.

    maxdist : float
        Distance in z-direction from the lowest point in the mesh to consider.

    Returns
    -------
    new_mesh : pyvista.PolyData
        Mesh with boundary downprojected.

    """

    lowest = mesh.bounds[int(flip)]
    boundary = get_boundary_edges(mesh)

    mpts = mesh.points
    bdpts = boundary.points
    idx_in_parent = np.array([mesh.FindPoint(ii) for ii in bdpts])

    to_adjust = idx_in_parent[bdpts[:, 0] - lowest < maxdist]
    mpts[to_adjust, 0] = lowest

    new_mesh = pyvista.PolyData(mpts, mesh.faces)

    return new_mesh

def downproject_border(mesh, value, axis=0, flip=False):
    """
    """

    lowest = value
    boundary = get_boundary_edges(mesh)

    mpts = mesh.points
    bdpts = boundary.points
    idx_in_parent = np.array([mesh.FindPoint(ii) for ii in bdpts])

    to_adjust = idx_in_parent[bdpts[:, 0] > value]
    mpts[to_adjust, axis] = lowest

    new_mesh = pyvista.PolyData(mpts, mesh.faces)

    return new_mesh


def get_boundary_points(mesh):
    """ Get indices of points in the boundary. """
    boundary = get_boundary_edges(mesh)
    bdpts = boundary.points
    indices = np.array([mesh.FindPoint(ii) for ii in bdpts])

    return indices


#def remesh(mesh, n_points, subratio=10, max_iter=10000, holefill=100):
#    """
#    Remesh the input PolyData.
#
#    Can be used to equalize the sizes of the triangles in the mesh.
#
#    Parameters
#    ----------
#    mesh : pyvista.PolyData
#        Mesh to operate on.
#
#    npoints : int
#        The number of vertices in the resulting mesh.
#
#    subratio : int, optional
#         TODO. Default = 10.
#
#    max_iter : int, optional
#        Maximal number of iterations before stopping. Default = 10000.
#
#    Returns
#    -------
#    new_mesh : pyvista.PolyData
#        Resulting mesh.
#
#    """
#    cobj = clustering.Cluster(mesh)
#    cobj.GenClusters(n_points, subratio=subratio, max_iter=max_iter)
#    cobj.GenMesh()
#
#    new_mesh = pyvista.PolyData(cobj.ReturnMesh())
#    new_mesh = ECFT(new_mesh, holefill)
#
#    return new_mesh

def remesh_decimate(mesh, iters, upfactor=2, downfactor=.5, verbose=True):
    """
    Iterative remeshing/decimation.

    Can be thought of as an alternative
    smoothing approach. The input mesh is remeshed with a factor times the
    original number of vertices, and then downsampled by another factor.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to operate on.

    iters : int
        Number of iterations.

    upfactor : float, optional
        Factor with which to upsample. Default = 2.

    downfactor : float, optional
        Factor with which to downsample. Default = 0.5.

    verbose : bool, optional
        Flag for whether to print operation steps. Default = True.

    Returns
    -------
    mesh : pyvista.PolyData
        Processed mesh.

    """
    for ii in range(iters):
        mesh = correct_bad_mesh(mesh, verbose=verbose)
        mesh = ECFT(mesh, 0)

        mesh = remesh(mesh, mesh.GetNumberOfPoints() * 2)
        mesh = mesh.compute_normals(inplace=False)
        mesh = mesh.decimate(.5, volume_preservation=True, normals=True, inplace=False)
        mesh = ECFT(mesh, 0)

    return mesh

def get_non_manifold_edges(mesh):
    """ Get non-manifold edges. """
    edges = mesh.extract_edges(boundary_edges=False,
                           non_manifold_edges=True, feature_edges=False,
                           manifold_edges=False)
    return edges

def get_boundary_edges(mesh):
    """ Get boundary edges. """
    edges = mesh.extract_edges(boundary_edges=True,
                           non_manifold_edges=False, feature_edges=False,
                           manifold_edges=False)
    return edges

def get_manifold_edges(mesh):
    """ Get manifold edges. """
    edges = mesh.extract_edges(boundary_edges=False,
                              non_manifold_edges=False, feature_edges=False,
                              manifold_edges=True)
    return edges

def get_feature_edges(mesh, angle=30):
    """ Get feature edges defined by given angle. """
    edges = mesh.extract_edges(feature_angle=angle, boundary_edges=False,
                              non_manifold_edges=False, feature_edges=True,
                              manifold_edges=False)
    return edges

def get_connected_vertices(mesh, index, include_self=True):

    connected_vertices = []
    if include_self:
        connected_vertices.append(index)

    cell_id_list = vtk.vtkIdList()
    mesh.GetPointCells(index, cell_id_list)

    # Loop through each cell using the seed point
    for ii in range(cell_id_list.GetNumberOfIds()):
        cell = mesh.GetCell(cell_id_list.GetId(ii))

        if cell.GetCellType() == 3:
            point_id_list = cell.GetPointIds()

            # add the point which isn't the seed
            to_add = point_id_list.GetId(1) if point_id_list.GetId(0) == index else point_id_list.GetId(0)
            connected_vertices.append(to_add)
        else:
            # Loop through the edges of the point and add all points on these.
            for jj in range(cell.GetNumberOfEdges()):
                point_id_list = cell.GetEdge(jj).GetPointIds()

                # add the point which isn't the seed
                to_add = point_id_list.GetId(1) if point_id_list.GetId(0) == index else point_id_list.GetId(0)
                connected_vertices.append(to_add)

    connected_vertices = np.unique(connected_vertices)

    return connected_vertices

def get_connected_vertices_all(mesh, include_self=True):

    connectivities = [[]] * mesh.n_points
    for ii in range(mesh.n_points):
        connectivities[ii] = get_connected_vertices(mesh, ii, include_self)

    connectivities = np.array(connectivities)

    return connectivities



def ECFT(mesh, hole_size=100.0, inplace=False):
    """
    Perform ExtractLargest, Clean, FillHoles, and TriFilter
    operations in sequence.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to operate on.

    holesize : float, optional
        Size of holes to fill. Default = 100.0.

    inplace : bool, optional
        Flag for performing operation in-place. Default = False.

    Returns
    -------
    new_mesh : pyvista.PolyData
        Mesh after operation. Returns None if inplace == True.

    """
    if inplace:
        new_mesh = mesh
    else:
        new_mesh = mesh.copy()

    new_mesh = new_mesh.extract_largest(inplace=False)
    new_mesh = new_mesh.clean(inplace=False)
    new_mesh = new_mesh.fill_holes(hole_size)
    new_mesh = new_mesh.tri_filter()

    if inplace:
        return
    else:
        return new_mesh

def define_meristem(mesh, pdata, method='central_mass', res=(1,1,1), fluo=None):
    """
    Determine which domain in the segmentation that corresponds to the meristem.
    Some methods are deprecated and should not be used.

    Parameters
    ----------
        mesh : pyvista.PolyData
        Mesh to operate on.

    pdata : pd.DataFrame
        Corresonding point data for input mesh.

    method : str, optional
        Method for defining the meristem to use. Default = 'central_mass'.

    res : 3-tuple, optional
        Resolution of the dimensions. Default = (1,1,1).

    fluo : np.ndarray, optional
        Intensity matrix.

    Returns
    -------
    meristem, ccoord : int, 3-tuple
        Domain index of the meristem, as well as the center coordinates using
        the given method.

    """
    # TODO: Sort out this function
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

    meristem = np.argmin(np.sqrt(np.sum((pdata[['z', 'y', 'x']] -
                                         ccoord)**2, axis=1)))
    meristem = pdata.loc[meristem, 'domain']
    return meristem, ccoord

def fit_paraboloid(data, init=[1, 1, 1, 1, 1, 0, 0, 0]):
    """
    Fit a paraboloid to arbitrarily oriented 3D data.

    The paraboloid data can by oriented along an arbitrary axis --
    not necessarily x, y, z. The function rotates the data points and returns
    the rotation angles along the x, y, z axis.

    Returns the parameters for a paraboloid along the z-axis. The angles can be
    used to correct the paraboloid for rotation.

    Paraboloid equation : p1 * x**2. + p2 * y**2. + p3 * x + p4 * y + p5 = z

    Parameters
    ----------
    data : np.ndarray
        Data to fit the paraboloid to.

    init : 8-tuple
        Initial parameters for the paraboloid.

    Returns
    -------
    popt : np.array
        Parameters after optimisation.

    """
    import scipy.optimize as opt
    from scipy.spatial.transform import Rotation as R
    from misc import rotate


    def errfunc(p, coord):
        p1, p2, p3, p4, p5, alpha, beta, gamma = p
        coord = rotate(coord, [alpha, beta, gamma])

        x, y, z = np.array(coord).T
        return abs(p1 * x**2. + p2 * y**2. + p3 * x + p4 * y + p5 - z)
    popt, _ = opt.leastsq(errfunc, init, args=(data,))

    return popt

def fit_paraboloid_mesh(mesh):
    """
    Fit a paraboloid to a mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to fit paraboloid to.

    Returns
    -------
    popt, apex : 8-tuple, 3-tuple
        Parameters for the paraboloid, as well as the coordinates for the
        paraboloid apex.

    """
    popt = fit_paraboloid(mesh.points, )
    apex = get_paraboloid_apex(popt)
    return popt, apex

def get_paraboloid_apex(p):
    """
    Return the apex coordinates of a paraboloid.

    Use the return of fit_paraboloid() to compute the apex of the paraboloid.
    The return is in the coordinate system of the data, meaning that the
    coordinates have been corrected for the rotation angles.

    Parameters
    ----------
    p : 8-tuple
        Parameters defining the paraboloid.

    Returns
    -------
    coords : np.array
        Coordinates for the paraboloid apex.

    """
    from misc import rotate

    p1, p2, p3, p4, p5, alpha, beta, gamma = p
    x = -p3 / (2. * p1)
    y = -p4 / (2. * p2)
    z = p1 * x**2. + p2 * y**2. + p3 * x + p4 * y + p5

    coords = rotate(np.array([[x, y, z], ]), [alpha, beta, gamma], True)[0]

    return coords
