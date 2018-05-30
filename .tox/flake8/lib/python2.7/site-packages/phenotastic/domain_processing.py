#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:38:43 2018

@author: henrik
"""
import numpy as np
import pandas as pd
#import networkx as nx
#import handy_functions as hf
import scipy
from vtk.util import numpy_support as nps
import vtk
import vtkInterface as vi
import copy
import Meristem_Phenotyper_3D as ap
from misc import merge
import misc

# TODO: Documentation
def domains_from_curvature(pdata):
    """ Create domains based on a Steepest Descent approach on an

    Parameters
    ----------
    pdata : pandas.DataFrame
        DataFrame containing information about mesh points, their curvature, and
        neighbourhood.

    Returns
    -------
    new_pdata : pandas.DataFrame
        DataFrame with domain label set.

    """
    new_pdata = pdata.copy()

    connections = []
    for key, value in enumerate(pdata['curv']):
        neighCurvs = pdata.loc[pdata['neighs'][key], 'curv']
        key_is_boa = all(neighCurvs < value)

        if key_is_boa:
            connections.append([key])
        else:
            connections.append([key, neighCurvs.idxmax()])

    doms = merge(connections)
    doms = np.array([np.array(list(doms[ii])) for ii in xrange(len(doms))])

    # Set domain values
    for ii in xrange(len(doms)):
        new_pdata.loc[doms[ii], 'domain'] = ii
    new_pdata.loc[:, 'domain'] = pd.Categorical(new_pdata.domain).codes

    return new_pdata

def get_boas(pdata):
    """ Returns which point indices in the input pdata are attractors, based on
    having the maximum curvature within the domain.

    Parameters
    ----------
    pdata : pandas.DataFrame


    Returns
    -------
    boas, bdata : np.ndarray, pandas.DataFrame
        Indices of boas and their corresponding data.

    """
    doms = np.unique(pdata.domain.values)
    doms = doms[~np.isnan(doms)]

    boas = np.array(
        [pdata.loc[pdata.domain == ii, 'curv'].idxmax() for ii in doms])
    bdata = pdata.iloc[boas]

    return boas, bdata

def merge_boas_distance(pdata, boas, bdata, distance):
    """ Merge basins of attraction based on their euclidian distance from one
    another.

    Uses a KDTree to find closest points.

    Parameters
    ----------
    pdata : pandas.DataFrame

    boas : np.ndarray

    bdata : pandas.DataFrame

    distance : float
        Maximal allowed euclidean distance. Smaller distances than this will be
        merged.

    Returns
    -------
    pdata : pandas.DataFrame
        New pdata object.

    """
    # Find BoAs within certain distance of each other
    tree = scipy.spatial.cKDTree(bdata[['x', 'y', 'z']])
    groups = tree.query_ball_point(bdata[['x', 'y', 'z']], distance)
    groups = misc.merge_lists_to_sets(groups)
    groups = np.array([np.array(ii) for ii in groups])

    # Merge domains
    for key, value in enumerate(groups):
        if len(value) > 1:
            newDomain = pdata.loc[boas[value[0]], 'domain']
            targetDomains = pdata.loc[boas[value[1:]], 'domain']
            pdata.loc[pdata['domain'].isin(
                targetDomains), 'domain'] = newDomain
    return pdata

# TODO: This piece of shit function
def merge_boas_engulfing(A, pdata, threshold=.9):
    """ Merge boas based on whether adjacent domains are encircling more than a
    certain fraction of the domain boundary.

    TODO: Unfinished function. Needs a rewrite.

    Parameters
    ----------
    A : Meristem_Phenotyper_3D.AutoPhenotype

    pdata : pandas.DataFrame

    threshold : float
        Minimum fraction enclosure needed in order to merge. (Default value = .9)

    Returns
    -------
    pdata : pandas.DataFrame

    """
    changed = True
    while changed:
        changed = False  # new round
        domains = pdata.domain.unique()

        # For every domain, find points facing other domains and domains facing
        # NaN
        for ii in domains:
            in_domain = np.where(pdata.domain == ii)[0]
            in_domain_boundary = np.intersect1d(in_domain,
                                                get_boundary_points(A.mesh))

            boundary_neighbours = np.unique(
                [x for y in pdata.loc[in_domain, 'neighs'].values for x in y])
            neighbouring_domains = pdata.loc[boundary_neighbours].loc[pdata.loc[boundary_neighbours,
                                                                 'domain'] != ii, 'domain']
            counts = neighbouring_domains.value_counts()
            frac = float(counts.max()) / (counts.sum() + len(in_domain_boundary))

            if frac > threshold:
                new_domain = neighbouring_domains.value_counts().idxmax()
                pdata.loc[pdata.domain == ii, 'domain'] = new_domain
                changed = True

    pdata.loc[:, 'domain'] = pd.Categorical(pdata.domain).codes
    return pdata


def merge_boas_depth(A, pdata, threshold=0.0, exclude_boundary=False, min_points=0):
    """ Merge domains based on their respective depths.

    Parameters
    ----------
    A :

    pdata :

    threshold :
         (Default value = 0.0)
    exclude_boundary :
         (Default value = False)
    min_points :
         (Default value = 0)

    Returns
    -------

    """
    boundary = get_boundary_points(A.mesh)
    changed = True
    while changed:
        changed = False  # new round
        domains = pdata.domain.unique()
        domains.sort()
        to_merge = []

        for ii in domains:
            in_domain = np.where(pdata.domain == ii)[0]
            max_curv = pdata.loc[in_domain, 'curv'].max()

            neighs_pts = np.array(
                [x for y in pdata.loc[in_domain, 'neighs'].values for x in y])
            # pts in neighbouring domains
            neighs_pts = np.array(
                filter(lambda x: x not in in_domain, neighs_pts))

            if exclude_boundary:
                in_domain = np.array(
                    filter(lambda x: x not in boundary, in_domain))
                neighs_pts = np.array(
                    filter(lambda x: x not in boundary, neighs_pts))

            # neighbouring domains
            neighs_doms = np.sort(pdata.loc[neighs_pts, 'domain'].unique())

            ''' Calculate average border curvature '''
            for jj in neighs_doms:
                # all the points in the neighbouring domain which has a neighbour in
                # the current domain
                border_pts = np.where(pdata.domain == jj)[0]
                border_pts = np.array(
                    filter(lambda x: x in neighs_pts, border_pts))

                # get neighbours of the neighbour's neighbours that are in the current
                # domain. Merge.
                border_pts_neighs = np.unique(
                    np.array([x for y in pdata.loc[border_pts, 'neighs'].values for x in y]))
                border_pts = np.append(
                    border_pts, [pt for pt in border_pts_neighs if pt in in_domain])
                border_pts = np.unique(border_pts)

                border_max_curv = pdata.loc[border_pts, 'curv'].max()

                # Only do if enough border
                if len(border_pts) < min_points:
                    continue

                # Merge
                if max_curv - border_max_curv < threshold:
                    to_merge.append([ii, jj])
                    print max_curv - border_max_curv
                    changed = True
                else:
                    to_merge.append([ii])
                    to_merge.append([jj])

        doms = merge(to_merge)
        domains_overwrite = copy.deepcopy(pdata.domain.values)
        for ii in xrange(len(doms)):
            domains_overwrite[pdata.domain.isin(list(doms[ii]))] = ii
        pdata.domain = domains_overwrite

    pdata.loc[:, 'domain'] = pd.Categorical(pdata.domain).codes
    return pdata

# TODO: Should be in a "misc" file


def merge_boas_border_curv(A, pdata, threshold=0.0, fct=np.mean, min_points=4,
                           exclude_boundary=False):
    """ Merge neighbouring domains based on their border curvature.

    Parameters
    ----------
    A :

    pdata :

    threshold :
         (Default value = 0.0)
    fct :
         (Default value = np.mean)
    min_points :
         (Default value = 4)
    exclude_boundary :
         (Default value = False)

    Returns
    -------

    """
    boundary = get_boundary_points(A.mesh)
    changed = True
    while changed:
        changed = False  # new round
        domains = pdata.domain.unique()
        domains.sort()
        to_merge = []

        for ii in domains:
            in_domain = np.where(pdata.domain == ii)[0]
            neighs_pts = np.array(
                [x for y in pdata.loc[in_domain, 'neighs'].values for x in y])
            # pts in neighbouring domains
            neighs_pts = np.array(
                filter(lambda x: x not in in_domain, neighs_pts))

            if exclude_boundary:
                in_domain = np.array(
                    filter(lambda x: x not in boundary, in_domain))
                neighs_pts = np.array(
                    filter(lambda x: x not in boundary, neighs_pts))

            # neighbouring domains
            neighs_doms = np.sort(pdata.loc[neighs_pts, 'domain'].unique())

            ''' Calculate average border curvature '''
            for jj in neighs_doms:
                # all the points in the neighbouring domain which has a neighbour in
                # the current domain
                border_pts = np.where(pdata.domain == jj)[0]
                border_pts = np.array(
                    filter(lambda x: x in neighs_pts, border_pts))

                # get neighbours of the neighbour's neighbours that are in the current
                # domain. Merge.
                border_pts_neighs = np.unique(
                    np.array([x for y in pdata.loc[border_pts, 'neighs'].values for x in y]))
                border_pts = np.append(
                    border_pts, [pt for pt in border_pts_neighs if pt in in_domain])
                border_pts = np.unique(border_pts)

                # Only do if enough border
                if len(border_pts) < min_points:
                    continue

                # Merge
                mean_border_curv = fct(pdata.loc[border_pts, 'curv'])
#        print mean_border_curv
                if mean_border_curv > threshold:
                    to_merge.append([ii, jj])
                    print mean_border_curv
                    changed = True
                else:
                    to_merge.append([ii])
                    to_merge.append([jj])

        import copy
        doms = merge(to_merge)
        domains_overwrite = copy.deepcopy(pdata.domain.values)
        for ii in xrange(len(doms)):
            domains_overwrite[pdata.domain.isin(list(doms[ii]))] = ii
        pdata.domain = domains_overwrite

    pdata.loc[:, 'domain'] = pd.Categorical(pdata.domain).codes
    return pdata


def get_boundary_points(mesh):
    """ Get point indices of mesh boundary.

    Parameters
    ----------
    mesh :


    Returns
    -------

    """
    fe = mesh.ExtractEdges(feature_angle=0, boundary_edges=True,
                           non_manifold_edges=False, manifold_edges=False,
                           feature_edges=False)

    # Get the coordinates for the respective points
    fepts = fe.points
    pts = mesh.points

    # Find the indices of the boundary points in the mesh points
    indices = [np.where(np.all(fepts[ii] == pts, axis=1))[0][0]
               for ii in xrange(len(fepts))]
    indices = np.sort(indices)

    return indices


def set_boundary_curv(curvs, mesh, value):
    """ Set the curvature of the mesh boundary.

    Parameters
    ----------
    curvs :

    mesh :

    value :


    Returns
    -------

    """
    newcurvs = curvs.copy()

    boundary = get_boundary_points(mesh)
    newcurvs[boundary] = value
    return newcurvs


def filter_curvature(curvs, neighs, fct, iters, exclude=[]):
    """ Filter curvature with a function. Exclude the list of indices if given.

    Parameters
    ----------
    curvs :

    neighs :

    fct :

    iters :

    exclude :
         (Default value = [])

    Returns
    -------

    """
    for ii in xrange(iters):
        new_curvs = copy.deepcopy(curvs)
        for jj in xrange(len(curvs)):
            val = np.nan
            to_proc = curvs[[kk for kk in neighs[jj] if kk not in exclude]]
            if len(to_proc) > 0:
                val = fct(to_proc)
            if not np.isnan(val):
                new_curvs[jj] = val
        curvs = new_curvs
    return curvs


def remove_boas_size(pdata, threshold, method="points"):
    """ Remove attractors mased on their size.

    Parameters
    ----------
    pdata :

    threshold :

    method :
         (Default value = "points")

    Returns
    -------

    """
    domain_sizes = pdata.loc[:, 'domain'].value_counts()
    if method == "points":
        to_remove = domain_sizes < threshold
    elif method == "relative_largest":
        to_remove = domain_sizes / domain_sizes.max() < threshold
    elif method == "relative_all":
        to_remove = domain_sizes / domain_sizes.sum() < threshold
    remove_domains = domain_sizes[to_remove].index.values
    pdata.loc[pdata.domain.isin(remove_domains), 'domain'] = np.nan
    return pdata


def nboas(pdata):
    """ Return the number of attractors.

    Parameters
    ----------
    pdata :


    Returns
    -------

    """
    return pd.DataFrame(np.array(pdata['domain']))[0].value_counts().size


def boas_npoints(pdata):
    """ Get the number of points per attractor.

    Parameters
    ----------
    pdata :


    Returns
    -------

    """
    return pd.DataFrame(np.array(pdata['domain']))[0].value_counts()

def init_pdata(A, curvs, neighs):
    """

    Parameters
    ----------
    A :

    curvs :

    neighs :


    Returns
    -------

    """
    coords = A.mesh.points
    domains = np.full((len(A.mesh.points), ), np.nan, dtype=np.float)
    pdata = pd.DataFrame(
        {'curv': curvs,
         'z': coords[:, 0],
         'y': coords[:, 1],
         'x': coords[:, 2],
         'domain': domains,
         'neighs': neighs
         })
    return pdata

def get_domain(mesh, pdata, domain):
    """ Get a domain from a labelled mesh.

    Parameters
    ----------
    mesh :

    pdata :

    domain :


    Returns
    -------

    """
    not_in_domain = pdata.loc[pdata.domain != domain].index.values
    mask = np.zeros((mesh.points.shape[0], ), dtype=np.bool)
    mask[not_in_domain] = True
    return vi.PolyData(mesh.RemovePoints(mask)[0])

def get_domain_boundary(mesh, pdata, domain):
    """ Get the point indices for a specified domain in a labelled mesh.

    Parameters
    ----------
    mesh :

    pdata :

    domain :


    Returns
    -------

    """
    dpoly = get_domain(mesh, pdata, domain)
    edges = dpoly.ExtractEdges(boundary_edges=True, manifold_edges=False,
                               feature_edges=False, non_manifold_edges=False)
    return edges


def define_meristem(mesh, pdata, method='central_mass', res=(0, 0, 0), fluo=None):
    """ Define which domain is the meristem.

    Parameters
    ----------
    mesh :

    pdata :

    method :
         (Default value = 'central_mass')
    res :
         (Default value = (0)
    fluo : np.ndarray
        Fluorescence data matrix. Needed for method == central_space.
        TODO: Avoid needing this.

    Returns
    -------
    meristem, coord : int, tuple
        Meristem domain index and coordinate tuple specifying the center
        coordinates.
    """
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

