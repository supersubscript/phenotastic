    #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 22:10:18 2018

@author: henrik
"""
import numpy as np
from vtk.util import numpy_support as nps
import vtkInterface as vi
import phenotastic.Meristem_Phenotyper_3D as ap
from PyACVD import Clustering
import vtk
from scipy.ndimage.morphology import binary_fill_holes
#import mesh_processing as mp

# Preprocessing
def fill_contour(contour, fill_xy=False, inplace=False):
    """
    Fill contour by closing all the edges (except for the top one), and applying
    a binary fill-holes operation. Note that this causes some errors if there is
    significant curvature on the contour, since the above-signal is
    down-projected. This can cause some erronous sharp edges which ruin the
    contour.

    Assumes first dimension being z, ordered bottom to top.

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

    """
    if not inplace:
        new_contour = contour.copy()
    else:
        new_contour = contour

    # Close all sides but top
    new_contour[0] = 1
    #new_contour[-1] = 1 # top
    new_contour[:, 0] = 1
    new_contour[:, -1] = 1
    new_contour[:, :, 0] = 1
    new_contour[:, :, -1] = 1

    # Fill holes form in xz & yz planes.
    for ii in xrange(new_contour.shape[1]):
        new_contour[:, ii] = binary_fill_holes(new_contour[:, ii])
    for ii in xrange(new_contour.shape[2]):
        new_contour[:, :, ii] = binary_fill_holes(new_contour[:, :, ii])

    # Remove edges again, also for top
    new_contour[0] = 0
    new_contour[-1] = 0
    new_contour[:, 0] = 0
    new_contour[:, -1] = 0
    new_contour[:, :, 0] = 0
    new_contour[:, :, -1] = 0

    if fill_xy:
        for ii in xrange(new_contour.shape[0]):
            new_contour[ii] = binary_fill_holes(new_contour[ii])

    new_contour = binary_fill_holes(new_contour)

    if not inplace:
        return new_contour
    else:
        return

### Actual mesh processing
def correct_bad_mesh(mesh, verbose=True):
    """
    Correct a bad (non-manifold) mesh with two methods:
        1) method removesmallcomponents from the pymeshfixpackage, and
        2) identifying leftover non-manifold edges and removing all the points
           in these.

    Parameters
    ----------
    mesh : vi.PolyData
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

    while nm.GetNumberOfPoints() > 0:
        if verbose:
            print('Trying to remove %d points' % nm.GetNumberOfPoints())

        # Create pymeshfix object from our mesh
        meshfix = _meshfix.PyTMesh()
        v, f = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        meshfix.LoadArray(v, f)

        # Remove smaller components
        meshfix.RemoveSmallestComponents()
        v2, f2 = meshfix.ReturnArrays()
        f2 = np.hstack([np.append(len(ii), ii) for ii in f2])

        # Create new polydata from cleaned out mesh
        new_poly = vi.PolyData(v2, f2)
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
        new_poly = new_poly.RemovePoints(mask)[0]

        new_poly = ECFT(new_poly, 0)
        nm = get_non_manifold_edges(new_poly)

    new_poly = ECFT(new_poly, 0)

    return new_poly

def remove_bridges(mesh, verbose=True):
    """
    Remove triangles where all vertices are part of the mesh.

    Parameters
    ----------
    mesh : vi.PolyData
        Mesh to operate on.

    verbose : bool, optional
        Flag to print processing steps.

    Notes
    -----
    Assumes triangulated mesh.

    Returns
    -------
    new_mesh : vi.PolyData
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
            print('Removing %d points' % len(remove_pts))
        if len(remove_pts) == 0:
            break

        # Actually remove
        mask = np.zeros((new_mesh.GetNumberOfPoints(),), dtype=np.bool)
        mask[remove_pts] = True

        new_mesh = new_mesh.RemovePoints(mask, keepscalars=False)[0]
        new_mesh = ECFT(new_mesh, 0)

    return new_mesh

def remove_normals(mesh, threshold_angle=0, flip=False):
    """ Remove points based on the point normal angle.

    Currently only considering the polar angle.

    Parameters
    ----------
    mesh : vi.PolyData
        Mesh on which to operate.

    threshold_angle : float, optional
        Threshold for the polar angle (theta). Values smaller than this will be
        removed. Default = 0.

    flip : bool, optional
         Flag to flip normal orientation. Default = False.

    Returns
    -------
    new_mesh : vi.PolyData
        Mesh with the resulting vertices removed.

    """
    normals = mesh.point_normals
    if flip:
        normals *= -1.
    normals = ap.cart2sphere(normals) / (2. * np.pi) * 360.
#    normals[:, 0] = 1.

    to_remove = normals[:, 1] < threshold_angle
    new_mesh = mesh.RemovePoints(to_remove, keepscalars=False)[0]
    return new_mesh

def geodesic(mesh, ii, jj):
    """
    Geodesic arc between two vertices on the input mesh.

    Utilises Dikstra's algorithm algorithm to compute this.

    Parameters
    ----------
    mesh : vi.PolyData
        Mesh to operate on.

    ii : int
        First point index.

    jj : int
        Second point index.

    Notes
    -----
    Requires manifold mesh.

    Returns
    -------
    arc : vi.PolyData
        PolyData corresponding to the geodesic line between the two points.

    """
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(mesh)
    dijkstra.SetStartVertex(ii)
    dijkstra.SetEndVertex(jj)
    dijkstra.Update()

    arc = dijkstra.GetOutput()
    arc = vi.PolyData(arc)

    return arc

def remove_tongues(mesh, radius, threshold=6, threshold2=0.8,
                   verbose=True):
    """
    Remove "tongues" in mesh.

    All boundary points within a given radius are considered. The ones where the
    fraction of the distance along the boundary, as divided by the euclidean
    distance, is greater than the given threshold. The second threshold compares
    the euclidean distance to the geodesic one, and removes the vertices in this
    if the threshold is met.

    Parameters
    ----------
    mesh : vi.PolyData
        Mesh to operate on.

    radius : float
        Radius for boundary point neighbourhood.

    threshold : float, optional
        Threshold for fraction between boundary distance and euclidean distance.
        Default = 6.

    threshold2 : float, optional
        Threshold for fraction between boundary distance and geodesic distance.
        Default = 0.8.

    Returns
    -------
    mesh : vi.PolyData
        Resulting mesh.

    """
    import networkx as nx
    from scipy.spatial import KDTree

    while True:
        # Preprocessing
        mesh = remove_bridges(mesh)
        mesh = ECFT(mesh, 0)

        # Get boundary information and index correspondences
        boundary = get_boundary_edges(mesh)
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

        # Remove points
        rmbool = np.zeros(mesh.points.shape[0], dtype=np.bool)
        rmbool[to_remove] = True

        mesh = mesh.RemovePoints(rmbool, keepscalars=False)[0]
        mesh = remove_bridges(mesh)
        mesh = correct_bad_mesh(mesh)
        mesh = ECFT(mesh, 0)

    return mesh

def drop_skirt(mesh, maxdist, flip=False):
    """
    Downprojects the boundary to the lowest point in the z-direction.

    Parameters
    ----------
    mesh : vi.PolyData
        Mesh to operate on.

    maxdist : float
        Distance in z-direction from the lowest point in the mesh to consider.

    Returns
    -------
    new_mesh : vi.PolyData
        Mesh with boundary downprojected.

    """

    lowest = mesh.GetBounds()[int(flip)]
    boundary = get_boundary_edges(mesh)

    mpts = mesh.points
    bdpts = boundary.points
    idx_in_parent = np.array([mesh.FindPoint(ii) for ii in bdpts])

    to_adjust = idx_in_parent[bdpts[:, 0] - lowest < maxdist]
    mpts[to_adjust, 0] = lowest

    new_mesh = vi.PolyData(mpts, mesh.faces)

    return new_mesh

def get_boundary_points(mesh):
    """ Get indices of points in the boundary. """
    boundary = get_boundary_edges(mesh)
    bdpts = boundary.points
    indices = np.array([mesh.FindPoint(ii) for ii in bdpts])

    return indices

def remesh(mesh, npoints, subratio=10, max_iter=10000, holefill=100):
    """
    Remesh the input PolyData.

    Can be used to equalize the sizes of the triangles in the mesh.

    Parameters
    ----------
    mesh : vi.PolyData
        Mesh to operate on.

    npoints : int
        The number of vertices in the resulting mesh.

    subratio : int, optional
         TODO. Default = 10.

    max_iter : int, optional
        Maximal number of iterations before stopping. Default = 10000.

    Returns
    -------
    new_mesh : vi.PolyData
        Resulting mesh.

    """
    cobj = Clustering.Cluster(mesh)
    cobj.GenClusters(npoints, subratio=subratio, max_iter=max_iter)
    cobj.GenMesh()

    new_mesh = vi.PolyData(cobj.ReturnMesh())
    new_mesh = ECFT(new_mesh, holefill)

    return new_mesh

def remesh_decimate(mesh, iters, upfactor=2, downfactor=.5, verbose=True):
    """
    Iterative remeshing/decimation.

    Can be thought of as an alternative
    smoothing approach. The input mesh is remeshed with a factor times the
    original number of vertices, and then downsampled by another factor.

    Parameters
    ----------
    mesh : vi.PolyData
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
    mesh : vi.PolyData
        Processed mesh.

    """
    for ii in xrange(iters):
        mesh = correct_bad_mesh(mesh, verbose=verbose)
        mesh = ECFT(mesh, 0)

        mesh = remesh(mesh, mesh.GetNumberOfPoints() * 2)
        mesh = mesh.GenerateNormals(inplace=False)
        mesh = mesh.Decimate(.5, volume_preservation=True, normals=True, inplace=False)
        mesh = ECFT(mesh, 0)

    return mesh

def get_non_manifold_edges(mesh):
    """ Get non-manifold edges. """
    edges = mesh.ExtractEdges(boundary_edges=False,
                           non_manifold_edges=True, feature_edges=False,
                           manifold_edges=False)
    return edges

def get_boundary_edges(mesh):
    """ Get boundary edges. """
    edges = mesh.ExtractEdges(boundary_edges=True,
                           non_manifold_edges=False, feature_edges=False,
                           manifold_edges=False)
    return edges

def get_manifold_edges(mesh):
    """ Get manifold edges. """
    edges = mesh.ExtractEdges(boundary_edges=False,
                              non_manifold_edges=False, feature_edges=False,
                              manifold_edges=True)
    return edges

def get_feature_edges(mesh, angle=30):
    """ Get feature edges defined by given angle. """
    edges = mesh.ExtractEdges(feature_angle=angle, boundary_edges=False,
                              non_manifold_edges=False, feature_edges=True,
                              manifold_edges=False)
    return edges

def ECFT(mesh, holesize=100.0, inplace=False):
    """
    Perform ExtractLargest, Clean, FillHoles, and TriFilter
    operations in sequence.

    Parameters
    ----------
    mesh : vi.PolyData
        Mesh to operate on.

    holesize : float, optional
        Size of holes to fill. Default = 100.0.

    inplace : bool, optional
        Flag for performing operation in-place. Default = False.

    Returns
    -------
    new_mesh : vi.PolyData
        Mesh after operation. Returns None if inplace == True.

    """
    if inplace:
        new_mesh = mesh
    else:
        new_mesh = mesh.Copy()

    new_mesh.ExtractLargest(inplace=True)
    new_mesh.Clean(inplace=True)
    new_mesh.FillHoles(holesize, inplace=True)
    new_mesh.TriFilter(inplace=True)

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
    mesh : vi.PolyData
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

    def errfunc(p, coord):
        p1, p2, p3, p4, p5, alpha, beta, gamma = p
        coord = ap.rot_coord(coord, [alpha, beta, gamma])
        x, y, z = np.array(coord).T
        return abs(p1 * x**2. + p2 * y**2. + p3 * x + p4 * y + p5 - z)
    popt, _ = opt.leastsq(errfunc, init, args=(data,))

    return popt

def fit_paraboloid_mesh(mesh):
    """
    Fit a paraboloid to a mesh.

    Parameters
    ----------
    mesh : vi.PolyData
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
    p1, p2, p3, p4, p5, alpha, beta, gamma = p
    x = -p3 / (2. * p1)
    y = -p4 / (2. * p2)
    z = p1 * x**2. + p2 * y**2. + p3 * x + p4 * y + p5
    coords = ap.rot_coord(np.array([[x, y, z], ]), [alpha, beta, gamma], True)[0]

    return coords
