#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:38:43 2018

@author: henrik
"""
import copy

import numpy as np
import pandas as pd
import pyvista
import scipy

from imgmisc import flatten, merge

import phenotastic.mesh as mp


def median(scalars, neighs=None, iterations=1):
    scalars = scalars.copy()
    for ii in range(iterations):
        scalars = filter_curvature(scalars, neighs, np.median, 1)
    return scalars


def minmax(scalars, neighs=None, iterations=1):
    scalars = scalars.copy()
    for ii in range(iterations):
        scalars = filter_curvature(scalars, neighs, np.min, 1)
        scalars = filter_curvature(scalars, neighs, np.max, 1)
    return scalars


def maxmin(scalars, neighs=None, iterations=1):
    scalars = scalars.copy()
    for ii in range(iterations):
        scalars = filter_curvature(scalars, neighs, np.max, 1)
        scalars = filter_curvature(scalars, neighs, np.min, 1)
    return scalars


# TODO: Documentation
def steepest_ascent(mesh, scalars, neighbours=None, verbose=False):
    """Create domains based on a Steepest Descent approach"""
    # Make checks and calculate neighbours if we don't have them.
    if len(scalars) != mesh.n_points or scalars.ndim > 1:
        raise RuntimeError("Invalid scalar array.")
    if neighbours is None:
        neighbours = mp.get_connected_vertices_all(mesh)

    # Get the individual connections by computing the neighbourhood gradients
    connections = [[]] * mesh.n_points
    for key, value in enumerate(scalars):
        neighbour_values = scalars[neighbours[key]]
        differences = neighbour_values - scalars[key]
        indices = neighbours[key][np.argmax(differences)]
        connections[key] = [key, indices]
    domains = merge(connections)
    domains = [np.array(list(domain)) for domain in domains]

    # Set domain values
    output = np.zeros(mesh.n_points, "int")
    for ii, domain_members in enumerate(domains):
        output[domain_members] = ii
    if verbose:
        print(f"Found {len(domains)} domains")

    return output


def relabel(domains, order):
    output = np.zeros(len(domains), "int")
    for ii, domain_members in enumerate(order):
        output[np.isin(domains, domain_members)] = ii
    return output


def map_to_domains(domains, values):
    doms = np.unique(domains)
    output = np.zeros(len(domains))
    for ii, domain_members in enumerate(doms):
        output[np.isin(domains, domain_members)] = values[ii]
    return output


def merge_angles(
    mesh, domains, meristem_index, threshold=20, method="center_of_mass", verbose=False
):

    #    mesh = A.mesh
    #    domains = A.mesh['domains']
    #    meristem_index = int(meristem_index)
    #    threshold = 12
    #    method='center_of_mass'
    #    verbose=False

    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))
    if n_domains_initial == 1:
        return domains

    changed = True
    while changed:

        if method in ["center_of_mass", "com"]:
            centers = [
                extract_domain(mesh, domains, ii).center_of_mass()
                for ii in np.unique(domains)
            ]
        else:
            raise RuntimeError('Method "{}" not valid.'.format(method))
        centers = np.array(centers)

        angles = np.array(
            [
                np.arctan2(
                    centers[ii][1] - centers[meristem_index][1],
                    centers[ii][2] - centers[meristem_index][2],
                )
                * 360.0
                / (2 * np.pi)
                % 360
                for ii in range(len(centers))
            ]
        )

        # reorder domains based on angles
        order = np.argsort(angles)
        new_domains = domains.copy()  # relabel(domains, order)
        #        angles = angles[order]

        indices = order.copy()
        indices = indices[1:]  # take out meristem
        angles = angles[indices]

        diffs = np.diff(angles, prepend=angles[-1]) % 360
        hits = np.where(diffs < threshold)[0]

        to_merge = [[meristem_index]] + [[ii] for ii in indices]
        for ii in hits:
            to_merge.append([indices[ii - 1], indices[(ii) % len(indices)]])

        domain_labels = merge(to_merge)
        domain_labels = np.array(
            [np.array(list(domain_labels[ii])) for ii in range(len(domain_labels))]
        )

        domains = relabel(new_domains, domain_labels)
        meristem_index = 0

        changed = True if len(domain_labels) < len(np.unique(new_domains)) else False

    # Set domain values
    if verbose:
        print(
            "Merging {} domains to {}.".format(
                n_domains_initial, len(np.unique(new_domains))
            )
        )
    output = domains  # relabel(mesh['domains'], domain_labels)
    return output


def merge_distance(
    mesh,
    domains,
    threshold,
    scalars=None,
    method="center_of_mass",
    metric="euclidean",
    verbose=False,
):
    """Merge basins of attraction based on their distance from one another.

    Uses a KDTree to find closest points.

    """
    method = method.lower()
    metric = metric.lower()
    n_domains_initial = len(np.unique(domains))

    # Define a method to use
    if method in ["center_of_mass", "com"]:
        coords = [
            extract_domain(mesh, domains, ii).center_of_mass()
            for ii in np.unique(domains)
        ]
    elif scalars is None:
        raise RuntimeError(
            'Method "{}" not viable without valid scalar input.'.format(method)
        )
    elif (
        method in ["maximum", "max", "minimum", "min"] and len(scalars) == mesh.n_points
    ):
        coords = []
        fct = (
            np.max
            if method in ["maximum", "max"]
            else np.min
            if method in ["minimum", "min"]
            else None
        )
        for ii in np.unique(domains):
            extremum = fct(scalars[domains == ii])
            index = np.where(np.logical_and(scalars == extremum, domains == ii))[0][0]
            coords.append(mesh.points[index])
    else:
        raise RuntimeError('Method "{}" not valid.'.format(method))
    coords = np.array(coords)

    # Find BoAs within certain distance of each other according to a given metric
    if metric == "euclidean":
        tree = scipy.spatial.cKDTree(coords)
        groups = tree.query_ball_point(coords, threshold)
    elif metric == "geodesic":
        indices = np.array([mesh.FindPoint(pt) for pt in coords])
        groups = []
        for ii, index1 in enumerate(indices):
            groups.append([ii])
            for jj, index2 in enumerate(indices):
                if mesh.geodesic_distance(index1, index2) < threshold:
                    groups[-1].append(jj)
    else:
        raise RuntimeError('Metric "{}" not valid.'.format(metric))
    groups = merge(groups)
    groups = np.array([np.array(list(ii)) for ii in groups])

    # Merge domains
    if verbose:
        print("Merging {} domains to {}.".format(n_domains_initial, len(groups)))
    output = relabel(domains, groups)

    return output


def extract_domain(mesh, domains, index):
    mesh = mesh.remove_points(domains != index)[0]
    return mesh


def neighbouring_domains(mesh, domains, seed, neighbours=None):
    """Get the neighbouring domains of a domain."""
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.get_connected_vertices_all(mesh)

    in_domain = np.where(domains == seed)[0]

    neighs_to_domain_boundary = np.unique(flatten(np.take(neighbours, in_domain)))
    neighbouring_domains = domains[neighs_to_domain_boundary][
        domains[neighs_to_domain_boundary] != seed
    ]
    neighbouring_domains = np.unique(neighbouring_domains)

    return neighbouring_domains


def border(mesh, domains, index1, index2, neighbours=None):
    if np.isin([index1, index2], domains).all():
        if index1 == index2:
            raise RuntimeError("index1 and index2 both {}".format(index1))
    else:
        raise RuntimeError("Indices must be in domains.")
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.get_connected_vertices_all(mesh)

    _, in_1 = get_domain_boundary(mesh, domains, index1, return_indices=True)
    _, in_2 = get_domain_boundary(mesh, domains, index2, return_indices=True)

    if in_1.shape[0] == 0 or in_2.shape[0] == 0:
        return []

    neighs_1 = flatten(np.take(neighbours, in_1))
    neighs_2 = flatten(np.take(neighbours, in_2))
    border = np.union1d(np.intersect1d(neighs_1, in_2), np.intersect1d(neighs_2, in_1))

    return border


def merge_engulfing(mesh, domains, threshold=0.9, neighbours=None, verbose=False):
    """Merge boas based on whether adjacent domains are encircling more than a
    certain fraction of the domain boundary.
    """
    # Make checks and calculate neighbours if we don't have them.
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.get_connected_vertices_all(mesh)

    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))

    changed = True
    while changed:
        changed = False

        # For every domain, find points facing other domains and domains facing NaN
        bidxs = boundary_indices(mesh)
        if len(bidxs) == 0:
            break

        for domain in np.unique(domains):
            in_domain = np.where(domains == domain)[0]

            in_domain_global_boundary = np.intersect1d(in_domain, bidxs)

            neighs_to_domain_boundary = np.unique(
                flatten(np.take(neighbours, in_domain))
            )
            neighbouring_domains = domains[neighs_to_domain_boundary][
                domains[neighs_to_domain_boundary] != domain
            ]
            neighbouring_domains, counts = np.unique(
                neighbouring_domains, return_counts=True
            )

            # calculate fraction of whole circumference bordering this neighbour
            border_frac = float(counts.max()) / (
                counts.sum() + len(in_domain_global_boundary)
            )

            # merge if appropriate
            if border_frac > threshold:
                new_domain = neighbouring_domains[np.argmax(counts)]
                domains[domains == domain] = new_domain
                changed = True

    if verbose:
        print(f"Merging {n_domains_initial} domains to {len(np.unique(domains))}")
    output = np.zeros(mesh.n_points, "float")
    for new_domain, old_domain in enumerate(np.unique(domains)):
        output[domains == old_domain] = new_domain

    return output


def merge_small(
    mesh,
    domains,
    threshold,
    metric="points",
    mode="border",
    neighbours=None,
    verbose=False,
):
    """Merge small domains to their neighbours."""
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.get_connected_vertices_all(mesh)

    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))

    changed = True
    while changed:
        changed = False  # new round
        d_labels, d_sizes = np.unique(domains, return_counts=True)
        to_merge = []

        if metric in ["points", "p", "point", "n_points", "npoints", "np"]:
            probes = d_labels[d_sizes < threshold]
        elif metric in ["points", "p", "point", "n_points", "npoints", "np"]:
            d_sizes = np.array(
                [extract_domain(mesh, domains, dd).area for dd in d_labels]
            )
            probes = d_labels[d_sizes < threshold]
            changed = True if len(probes) > 0 else False

        for probe in probes:
            probe_d_neighbours = neighbouring_domains(
                mesh, domains, probe, neighbours=neighbours
            )
            if mode == "border":
                d_borders = [
                    border(mesh, domains, probe, ii, neighbours=neighbours)
                    for ii in probe_d_neighbours
                ]
                d_border_sizes = [len(bb) for bb in d_borders]
                to_merge.append([probe, probe_d_neighbours[np.argmax(d_border_sizes)]])
            elif mode == "area":
                d_neighbour_areas = [
                    extract_domain(mesh, domains, pp).area for pp in probe_d_neighbours
                ]
                to_merge.append(
                    [probe, probe_d_neighbours[np.argmax(d_neighbour_areas)]]
                )

        if changed:
            doms = merge(to_merge)
            domains_overwrite = domains.copy()
            for ii in range(len(doms)):
                domains_overwrite[np.isin(domains, list(doms[ii]))] = ii

            domains = domains_overwrite

    if verbose:
        print(
            "Merging {} domains to {}.".format(
                n_domains_initial, len(np.unique(domains))
            )
        )
    output = np.zeros(mesh.n_points, "float")
    for new_domain, old_domain in enumerate(np.unique(domains)):
        output[domains == old_domain] = new_domain

    return output


def merge_disconnected(
    mesh, domains, meristem_index, threshold, neighbours=None, verbose=False
):
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.get_connected_vertices_all(mesh)
    if threshold is None:
        return domains

    meristem_index = int(meristem_index)
    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))

    _, meristem_boundary = get_domain_boundary(
        mesh, domains, meristem_index, return_indices=True
    )

    changed = True
    while changed:
        changed = False  # new round
        d_labels = np.unique(domains)

        # Get all borders to meristem. Figure out which are disconnected
        borders = [
            border(mesh, domains, meristem_index, ii, neighbours=neighbours)
            for ii in d_labels[d_labels != meristem_index]
        ]
        mask = np.array([len(borders[ii]) for ii in range(len(borders))]) == 0
        to_merge = [[meristem_index]] + [
            [ii] for ii in d_labels[d_labels != meristem_index][np.logical_not(mask)]
        ]
        probes = d_labels[d_labels != meristem_index][mask]
        meristem_index = 0.0

        # Merge with neighbours with most vertices in the corresponding border
        for probe in np.sort(probes):
            probe_borders = [
                border(mesh, domains, probe, jj, neighbours=neighbours)
                for jj in d_labels[d_labels != probe]
            ]
            border_sizes = [len(jj) for jj in probe_borders]
            connected_neighbour = d_labels[d_labels != probe][np.argmax(border_sizes)]

            to_merge.append([probe, connected_neighbour])
            changed = True

        if changed:
            doms = merge(to_merge)
            domains_overwrite = domains.copy()
            for ii in range(len(doms)):
                domains_overwrite[np.isin(domains, list(doms[ii]))] = ii

            domains = domains_overwrite

    if verbose:
        print(
            "Merging {} domains to {}.".format(
                n_domains_initial, len(np.unique(domains))
            )
        )
    output = np.zeros(mesh.n_points, "float")
    for new_domain, old_domain in enumerate(np.unique(domains)):
        output[domains == old_domain] = new_domain

    return output


# def merge_boas_disconnected(A, pdata, meristem, threshold=.9, threshold2=0.2, **kwargs):
#    """
#    TODO: REWRITE
#    Merge boas that have 1) less then threshold2 of a fraction of its boundary
#    points not connected to the meristem, and 2) has a neighbour which borders
#    at least a fraction of the given threshold of the domain's border vertices.
#    """
#    changed = True
#    while changed:
#        changed = False  # new round
#        domains = pdata.domain.unique()
#        domains = domains[domains != meristem]
#
#        # For every domain, find points facing other domains and domains facing
#        # NaN
#        for ii in domains:
##            print ii
#            in_domain = np.where(pdata.domain == ii)[0]
#            in_domain_boundary = np.array(
#                    [A.mesh.FindPoint(jj) for jj in
#                     get_domain_boundary(A.mesh, pdata, ii).points])
#
#            #np.intersect1d(in_domain,
#                                  #              get_boundary_points(A.mesh))
#
#            boundary_neighbours = np.unique(
#                [x for y in pdata.loc[in_domain, 'neighs'].values for x in y])
#            neighbouring_domains = pdata.loc[boundary_neighbours].loc[pdata.loc[boundary_neighbours,
#                                                                 'domain'] != ii, 'domain']
#            counts = neighbouring_domains.value_counts()
#
#            # TODO: So tired when I wrote this shit
#            too_small = False
#            if meristem in counts.index:
#                 too_small = float(counts.loc[meristem]) / len(in_domain_boundary) < threshold2
#
#            if (meristem not in counts.index.values) or too_small:
#                frac = float(counts.max()) / (counts.sum() + len(in_domain_boundary))
#
#                if frac > threshold:
#                    new_domain = neighbouring_domains.value_counts().idxmax()
#                    pdata.loc[pdata.domain == ii, 'domain'] = new_domain
#                    changed = True
#
#    pdata.loc[:, 'domain'] = pd.Categorical(pdata.domain).codes
#    return pdata


def merge_depth(
    mesh,
    domains,
    scalars,
    threshold=0.0,
    neighbours=None,
    exclude_boundary=False,
    min_points=0,
    mode="max",
    verbose=False,
):
    """Merge domains based on their respective depths."""
    # mesh, domains=mesh['domains'], scalars=mesh['curvature'], neighbours=neighs, threshold=0.02, verbose=True
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domain array.")
    if len(scalars) != mesh.n_points or scalars.ndim > 1:
        raise RuntimeError("Invalid scalar array.")
    if neighbours is None:
        neighbours = mp.get_connected_vertices_all(mesh)

    domains = domains.copy()
    boundary = boundary_indices(mesh)
    n_domains_initial = np.unique(domains).shape[0]

    if mode == "min":
        fct = np.min
    elif mode == "median":
        fct = np.median
    elif mode == "max":
        fct = np.max
    else:
        fct = np.mean

    changed = True
    while changed:
        changed = False
        to_merge = []

        for dom in np.unique(domains):
            in_domain = np.where(domains == dom)[0]
            max_value = np.max(scalars[in_domain])

            # get the points that are in neighbouring domains
            neighs_pts = [x for y in np.take(neighbours, in_domain) for x in y]
            neighs_pts = [x for x in neighs_pts if x not in in_domain]

            if exclude_boundary:
                in_domain = [x for x in in_domain if x not in boundary]
                neighs_pts = [x for x in neighs_pts if x not in boundary]

            # neighbouring domains, in order
            neighs_doms = np.unique(domains[neighs_pts])
            neighs_doms = np.sort(neighs_doms)

            for neigh_dom in neighs_doms:
                # all the points in the neighbouring domain which has a neighbour in
                # the current domain
                border_pts = np.where(domains == neigh_dom)[0]
                border_pts = np.array([x for x in border_pts if x in neighs_pts])

                # get neighbours of the neighbour's neighbours that are in the current
                # domain. Merge.
                border_pts_neighs = np.unique(
                    np.array([x for y in np.take(neighbours, border_pts) for x in y])
                )
                border_pts = np.append(
                    border_pts, [pt for pt in border_pts_neighs if pt in in_domain]
                )
                border_pts = np.unique(border_pts)

                border_max_value = fct(scalars[border_pts])

                # p = pv.Plotter(notebook=False)
                # p.add_mesh(mesh, scalars='domains', categories=True, cmap='glasbey', interpolate_before_map=False)
                # p.add_points(mesh.points[border_pts], color='red')
                # p.add_points(mesh.points[border_pts[[np.argmax(scalars[border_pts])]]], color='blue', point_size=10)
                # p.add_points(mesh.points[border_pts[[np.argmax(scalars[border_pts])]]], color='blue', point_size=10)
                # p.add_points(mesh.points[in_domain[[np.argmax(scalars[in_domain])]]], color='blue', point_size=10)
                # scalars[in_domain]
                # p.show()

                # Only do if enough border
                if len(border_pts) < min_points:
                    continue

                # Merge
                if max_value - border_max_value < threshold:
                    to_merge.append([dom, neigh_dom])
                    changed = True
                else:
                    to_merge.append([dom])
                    to_merge.append([neigh_dom])

        # Update domains
        doms = merge(to_merge)
        domains_overwrite = domains.copy()
        for ii, dom in enumerate(doms):
            domains_overwrite[np.isin(domains, list(dom))] = ii
        domains = domains_overwrite

        if len(np.unique(domains)):
            break

    # Relabel between 0 and len(domains)
    if verbose:
        print(
            f"Merging {n_domains_initial} domains to {len(np.unique(domains))} domains"
        )
    new_domains = domains.copy()
    for ii, domain in enumerate(np.unique(domains)):
        new_domains[domains == domain] = ii

    return new_domains


def merge_border_length(mesh, domains, threshold=0.0, neighbours=None, verbose=False):
    """Merge domains based on their respective border lengths."""
    # mesh, domains=mesh['domains'], scalars=mesh['curvature'], neighbours=neighs, threshold=0.02, verbose=True
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domain array.")
    if neighbours is None:
        neighbours = mp.get_connected_vertices_all(mesh)

    domains = domains.copy()
    n_domains_initial = np.unique(domains).shape[0]

    changed = True
    while changed:
        changed = False
        to_merge = []

        for dom in np.unique(domains):
            in_domain = np.where(domains == dom)[0]

            # get the points that are in neighbouring domains
            neighs_pts = [x for y in np.take(neighbours, in_domain) for x in y]
            neighs_pts = [x for x in neighs_pts if x not in in_domain]

            # neighbouring domains, in order
            neighs_doms = np.unique(domains[neighs_pts])
            neighs_doms = np.sort(neighs_doms)

            for neigh_dom in neighs_doms:
                # all the points in the neighbouring domain which has a neighbour in
                # the current domain
                border_pts = np.where(domains == neigh_dom)[0]
                border_pts = np.array([x for x in border_pts if x in neighs_pts])

                # get neighbours of the neighbour's neighbours that are in the current
                # domain. Merge.
                border_pts_neighs = np.unique(
                    np.array([x for y in np.take(neighbours, border_pts) for x in y])
                )
                border_pts = np.append(
                    border_pts, [pt for pt in border_pts_neighs if pt in in_domain]
                )
                border_pts = np.unique(border_pts)

                # p = pv.Plotter(notebook=False)
                # p.add_mesh(mesh, scalars='domains', categories=True, cmap='glasbey', interpolate_before_map=False)
                # p.add_points(mesh.points[border_pts], color='red')
                # p.add_points(mesh.points[border_pts[[np.argmax(scalars[border_pts])]]], color='blue', point_size=10)
                # p.add_points(mesh.points[border_pts[[np.argmax(scalars[border_pts])]]], color='blue', point_size=10)
                # p.add_points(mesh.points[in_domain[[np.argmax(scalars[in_domain])]]], color='blue', point_size=10)
                # scalars[in_domain]
                # p.show()

                # Merge
                if len(border_pts) > threshold:
                    to_merge.append([dom, neigh_dom])
                    changed = True
                else:
                    to_merge.append([dom])
                    to_merge.append([neigh_dom])

        # Update domains
        doms = merge(to_merge)
        domains_overwrite = domains.copy()
        for ii, dom in enumerate(doms):
            domains_overwrite[np.isin(domains, list(dom))] = ii
        domains = domains_overwrite

        if len(np.unique(domains)):
            break

    # Relabel between 0 and len(domains)
    if verbose:
        print(
            f"Merging {n_domains_initial} domains to {len(np.unique(domains))} domains"
        )
    new_domains = domains.copy()
    for ii, domain in enumerate(np.unique(domains)):
        new_domains[domains == domain] = ii

    return new_domains


# TODO: Should be in a "misc" file
def merge_boas_border_curv(
    A, pdata, threshold=0.0, fct=np.mean, min_points=4, exclude_boundary=False, **kwargs
):
    """Merge neighbouring domains based on their border curvature.

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
    boundary = boundary_indices(A.mesh)
    changed = True
    while changed:
        changed = False  # new round
        domains = pdata.domain.unique()
        domains.sort()
        to_merge = []

        for ii in domains:
            in_domain = np.where(pdata.domain == ii)[0]
            neighs_pts = np.array(
                [x for y in pdata.loc[in_domain, "neighs"].values for x in y]
            )
            # pts in neighbouring domains
            neighs_pts = np.array([x for x in neighs_pts if x not in in_domain])

            if exclude_boundary:
                in_domain = np.array([x for x in in_domain if x not in boundary])
                neighs_pts = np.array([x for x in neighs_pts if x not in boundary])

            # neighbouring domains
            neighs_doms = np.sort(pdata.loc[neighs_pts, "domain"].unique())

            """ Calculate average border curvature """
            for jj in neighs_doms:
                # all the points in the neighbouring domain which has a neighbour in
                # the current domain
                border_pts = np.where(pdata.domain == jj)[0]
                border_pts = np.array([x for x in border_pts if x in neighs_pts])

                # get neighbours of the neighbour's neighbours that are in the current
                # domain. Merge.
                border_pts_neighs = np.unique(
                    np.array(
                        [x for y in pdata.loc[border_pts, "neighs"].values for x in y]
                    )
                )
                border_pts = np.append(
                    border_pts, [pt for pt in border_pts_neighs if pt in in_domain]
                )
                border_pts = np.unique(border_pts)

                # Only do if enough border
                if len(border_pts) < min_points:
                    continue

                # Merge
                mean_border_curv = fct(pdata.loc[border_pts, "curv"])
                #        print mean_border_curv
                if mean_border_curv > threshold:
                    to_merge.append([ii, jj])
                    print(mean_border_curv)
                    changed = True
                else:
                    to_merge.append([ii])
                    to_merge.append([jj])

        import copy

        doms = merge(to_merge)
        domains_overwrite = copy.deepcopy(pdata.domain.values)
        for ii in range(len(doms)):
            domains_overwrite[pdata.domain.isin(list(doms[ii]))] = ii
        pdata.domain = domains_overwrite

    pdata.loc[:, "domain"] = pd.Categorical(pdata.domain).codes
    return pdata


def boundary_indices(mesh, **kwargs):
    """Get point indices of mesh boundary."""
    fe = mesh.extract_feature_edges(
        feature_angle=0,
        boundary_edges=True,
        non_manifold_edges=False,
        manifold_edges=False,
        feature_edges=False,
    )

    # Get the coordinates for the respective points
    fepts = fe.points
    pts = mesh.points

    # Find the indices of the boundary points in the mesh points
    indices = [
        np.where(np.all(fepts[ii] == pts, axis=1))[0][0] for ii in range(len(fepts))
    ]
    indices = np.sort(indices)

    return indices


def set_boundary_values(mesh, scalars, values):
    """Set the curvature of the mesh boundary."""
    new_scalars = scalars.copy()
    boundary = boundary_indices(mesh)
    if len(boundary) > 0:
        new_scalars[boundary] = values
    return new_scalars


def filter_curvature(curvs, neighs, fct, iters, exclude=[], **kwargs):
    """Filter curvature with a function. Exclude the list of indices if given.
    Returns
    -------


    """
    for ii in range(iters):
        new_curvs = copy.deepcopy(curvs)
        for jj in range(len(curvs)):
            val = np.nan
            to_proc = curvs[[kk for kk in neighs[jj] if kk not in exclude]]
            if len(to_proc) > 0:
                val = fct(to_proc)
            if not np.isnan(val):
                new_curvs[jj] = val
        curvs = new_curvs
    return curvs


def mean(scalars, neighs, iters, exclude=[]):
    return filter(scalars, neighs, np.mean, iters, exclude)


def filter(scalars, neighs, fct, iters, exclude=[], **kwargs):
    """Filter curvature with a function. Exclude the list of indices if given."""
    for ii in range(iters):
        new_scalars = copy.deepcopy(scalars)
        for jj in range(len(scalars)):
            val = np.nan
            to_proc = scalars[[kk for kk in neighs[jj] if kk not in exclude]]
            if len(to_proc) > 0:
                val = fct(to_proc)
            if not np.isnan(val):
                new_scalars[jj] = val
        scalars = new_scalars
    return scalars


def remove_size(mesh, domains, threshold, method="points", relative="largest"):
    """Remove attractors based on their size."""
    method = method.lower()
    relative = relative.lower()

    # What's the metric?
    if method in ["points", "point", "p"]:
        groups, sizes = np.unique(domains, return_counts=True)
    elif method in ["area", "a"]:
        groups = np.unique(domains)
        sizes = np.array([extract_domain(mesh, domains, dd).area for dd in groups])
    else:
        raise RuntimeError("Invalid method.")

    # What are we comparing against?
    if relative == "all":
        reference = np.sum(sizes)
    elif relative == "largest":
        reference = np.max(sizes)
    elif relative in ["absolute", "abs"]:
        reference = 1
    else:
        raise RuntimeError("Invalid comparison option.")

    to_remove = np.where(sizes / reference < threshold)[0]
    to_remove = groups[to_remove]
    to_remove = np.isin(domains, to_remove)

    output = mesh.remove_points(to_remove, keep_scalars=True)[0]

    return output


def get_domain(mesh, pdata, domain, **kwargs):
    """Get a domain from a labelled mesh.

    Parameters
    ----------
    mesh :

    pdata :

    domain :


    Returns
    -------


    """
    not_in_domain = pdata.loc[pdata.domain != domain].index.values
    mask = np.zeros((mesh.points.shape[0],), dtype=np.bool)
    mask[not_in_domain] = True
    return pyvista.pointset.PolyData(mesh.remove_points(mask)[0])


def get_domain_boundary(mesh, domains, index, return_indices=False):
    """Get the point indices for a specified domain in a labelled mesh."""
    dpoly = extract_domain(mesh, domains, index)
    edges = dpoly.extract_feature_edges(
        boundary_edges=True,
        manifold_edges=False,
        feature_edges=False,
        non_manifold_edges=False,
    )

    if return_indices:
        indices = np.array([mesh.FindPoint(pt) for pt in edges.points])
        return edges, indices
    else:
        return edges


def domain_neighbors(mesh, domains, neighs):
    doms = [extract_domain(mesh, domains, dd) for dd in np.unique(domains)]
    dom_boundaries = [boundary_indices(dd) for dd in doms]
    doms_orig_indices = []
    for ii, dom in enumerate(doms):
        orig = [mesh.FindPoint(pt) for pt in dom.points[dom_boundaries[ii]]]
        doms_orig_indices.append(orig)

    neighs = np.array(neighs.copy())
    n_neighs = []
    for dom_orig_indices in doms_orig_indices:
        dom_neighs = flatten([domains[dd] for dd in neighs[dom_orig_indices]])
        dom_neighs = np.unique(dom_neighs)
        n_neighs.append(len(dom_neighs) - 1)
    return n_neighs
    # dom_boundaries = [[dom.FindPoint(pt) for pt in dom.points[dd]] for dd in ]


def define_meristem(
    mesh, domains, method="center_of_mass", return_coordinates=False, neighs=None
):
    """Define which domain is the meristem."""
    method = method.lower()

    if method in ["center_of_mass", "com"]:
        coord = mesh.center_of_mass()
    elif method in ["center", "c", "bounds"]:
        coord = np.mean(np.reshape(mesh.bounds, (3, -1)), axis=1)
    elif method in ["n_neighs", "neighbors", "neighs", "n_neighbors"]:
        doms = np.unique(domains)
        n_neighs = domain_neighbors(mesh, domains, neighs)
        meristem = doms[np.argmax(n_neighs)]
        coord = extract_domain(mesh, domains, meristem).center_of_mass()

    meristem = int(domains[mesh.FindPoint(coord)])

    if return_coordinates:
        return meristem, coord
    else:
        return meristem


def extract_domaindata(pdata, mesh, apex, meristem, **kwargs):
    """

    Parameters
    ----------
    pdata :

    mesh :

    apex :

    meristem :


    Returns
    -------

    """
    domains = np.unique(pdata.domain)
    domains = domains[~np.isnan(domains)]
    ddata = pd.DataFrame(
        columns=[
            "domain",
            "dist_boundary",
            "dist_com",
            "angle",
            "area",
            "maxdiam",
            "maxdiam_xy",
            "com",
            "ismeristem",
        ],
        dtype=np.object,
    )

    for ii in domains:
        # Get distance for closest boundary point to apex
        dom = get_domain(mesh, pdata, ii)
        dom_boundary = boundary_indices(dom)
        dom_boundary_coords = dom.points[dom_boundary]
        dom_boundary_dists = np.sqrt(np.sum((dom_boundary_coords - apex) ** 2, axis=1))
        d2boundary = np.min(dom_boundary_dists)

        # Get distance for center of mass from apex
        center = dom.center_of_mass()
        d2com = np.sqrt(np.sum((center - apex) ** 2))

        # Get domain size in terms of bounding boxes
        #        bounds = dom.GetBounds()
        domcoords = dom.points
        from scipy.spatial.distance import cdist

        dists = cdist(domcoords, domcoords)
        maxdiam = np.max(dists)
        dists_xy = cdist(domcoords[:, 1:], domcoords[:, 1:])
        maxdiam_xy = np.max(dists_xy)

        # Get domain angle in relation to apex
        rel_pos = center - apex
        angle = np.arctan2(rel_pos[1], rel_pos[2])  # angle in yz-plane
        if angle < 0:
            angle += 2.0 * np.pi
        angle *= 360 / (2.0 * np.pi)

        # Get surface area
        area = dom.area

        # Define type
        ismeristem = ii == meristem
        if ismeristem:
            angle = np.nan

        # Set data
        ddata.loc[int(ii)] = [
            int(ii),
            d2boundary,
            d2com,
            angle,
            area,
            maxdiam,
            maxdiam_xy,
            tuple(center),
            ismeristem,
        ]
    ddata = ddata.infer_objects()
    ddata = ddata.sort_values(["ismeristem", "area"], ascending=False)
    return ddata


# def merge_boas_angle(pdata, ddata, mesh, threshold, apex, **kwargs):
#    """Merge domains based on the angles between them.
#
#    Parameters
#    ----------
#    pdata :
#
#    ddata :
#
#    mesh :
#
#    threshold :
#
#    apex :
#
#
#    Returns
#    -------
#
#
#    """
#
##    boundary = get_boundary_points(mesh)
#    new_ddata = ddata.copy()
#    new_pdata = pdata.copy()
#    apex = apex.copy()
##    meristem_index = copy.copy(meristem_index)
#
##    meristem = new_ddata.loc[new_ddata.ismeristem].copy()
##    new_ddata = new_ddata.loc[~np.isnan(new_ddata.angle)].copy()
#
#    changed = True
#    while changed:
#        changed = False  # new round
#        new_ddata = new_ddata.sort_values('angle', na_position='first')
#        angles = np.sort(new_ddata[1:].angle.values)
#        domains = new_ddata[1:].domain.values.astype(np.int)
#
#        diffs = np.diff(np.append(angles, angles[0]))
#        diffs[diffs < 0] += 360
#
#        too_close = np.where(diffs < threshold)[0]
#
#        if len(too_close) > 0:
#            changed = True
#        else:
#            break
#
#        to_merge = [[domains[ii], domains[(ii + 1) % len(domains)]] for ii in too_close]
#
#        flatten = lambda l: [item for sublist in l for item in sublist]
#        to_merge.extend([[ii] for ii in domains if ii not in flatten(to_merge)])
#        to_merge.insert(0, [0]) # meristem
#
#        doms = merge(to_merge)
#        domains_overwrite = copy.deepcopy(new_pdata.domain.values)
#        for ii in range(len(doms)):
#            domains_overwrite[new_pdata.domain.isin(list(doms[ii]))] = ii
#        new_pdata.domain = domains_overwrite
#
#        new_ddata = extract_domaindata(new_pdata, mesh, apex, 0)
#        new_pdata, new_ddata = relabel_domains(new_pdata, new_ddata)
#        new_ddata = new_ddata.sort_values(['ismeristem', 'area'], ascending=False)
#
#    new_ddata = new_ddata.sort_values(['ismeristem', 'area'], ascending=False)
#    return new_pdata, new_ddata


def relabel_domains(pdata, ddata, order="area", **kwargs):
    """

    Parameters
    ----------
    pdata :

    ddata :

    order :
         (Default value = 'area')

    Returns
    -------

    """
    new_pdata = pdata.copy()
    new_ddata = ddata.copy()

    if order == "area":
        new_ddata.sort_values(["ismeristem", "area"], ascending=False)
    elif order == "maxdiam":
        new_ddata.sort_values(["ismeristem", "maxdiam"], ascending=False)
    elif order == "maxdiam_xy":
        new_ddata.sort_values(["ismeristem", "maxdiam_xy"], ascending=False)

    dmap = dict()
    for ii in range(len(new_ddata)):
        old_dom = new_ddata.iloc[ii].domain
        dmap[old_dom] = ii
        new_ddata["domain"].iloc[ii] = ii

    for ii in dmap:
        new_pdata.loc[pdata.domain == ii, "domain"] = dmap[ii]

    return new_pdata, new_ddata
