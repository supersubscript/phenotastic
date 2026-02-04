import copy
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
import scipy
from imgmisc import flatten, merge
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

import phenotastic.mesh as mp


def median(
    scalars: NDArray[np.floating[Any]],
    neighs: list[NDArray[np.intp]] | None = None,
    iterations: int = 1,
) -> NDArray[np.floating[Any]]:
    """Apply median filter to mesh-based scalar arrays.

    Args:
        scalars: Scalar array associated with mesh vertices
        neighs: Neighbor connectivity array for each vertex
        iterations: Number of filter iterations to apply

    Returns:
        Filtered scalar array
    """

    scalars = scalars.copy()
    for _ii in range(iterations):
        scalars = filter_curvature(scalars, neighs, np.median, 1)
    return scalars


def minmax(
    scalars: NDArray[np.floating[Any]],
    neighs: list[NDArray[np.intp]] | None = None,
    iterations: int = 1,
) -> NDArray[np.floating[Any]]:
    """Apply min-max filter to mesh-based scalar arrays.

    Args:
        scalars: Scalar array associated with mesh vertices
        neighs: Neighbor connectivity array for each vertex
        iterations: Number of filter iterations to apply

    Returns:
        Filtered scalar array
    """

    scalars = scalars.copy()
    for _ii in range(iterations):
        scalars = filter_curvature(scalars, neighs, np.min, 1)
        scalars = filter_curvature(scalars, neighs, np.max, 1)
    return scalars


def maxmin(
    scalars: NDArray[np.floating[Any]],
    neighs: list[NDArray[np.intp]] | None = None,
    iterations: int = 1,
) -> NDArray[np.floating[Any]]:
    """Apply max-min filter to mesh-based scalar arrays.

    Args:
        scalars: Scalar array associated with mesh vertices
        neighs: Neighbor connectivity array for each vertex
        iterations: Number of filter iterations to apply

    Returns:
        Filtered scalar array
    """

    scalars = scalars.copy()
    for _ii in range(iterations):
        scalars = filter_curvature(scalars, neighs, np.max, 1)
        scalars = filter_curvature(scalars, neighs, np.min, 1)
    return scalars


def steepest_ascent(
    mesh: pv.PolyData,
    scalars: NDArray[np.floating[Any]],
    neighbours: list[NDArray[np.intp]] | None = None,
    verbose: bool = False,
) -> NDArray[np.integer[Any]]:
    """Create domains using steepest ascent approach.

    Connects vertices based on the steepest local gradient in the scalar field.

    Args:
        mesh: PyVista mesh to create domains for
        scalars: 1D scalar array with length matching mesh.n_points
        neighbours: Neighbor connectivity array for each vertex
        verbose: Print information about domain creation process

    Returns:
        Array of domain labels with length mesh.n_points

    Raises:
        RuntimeError: If scalar array has invalid dimensions
    """

    # Make checks and calculate neighbours if we don't have them.
    if (len(scalars) != mesh.n_points) or scalars.ndim > 1:
        raise RuntimeError("Invalid scalar array.")
    if neighbours is None:
        neighbours = mp.vertex_neighbors_all(mesh)

    # Get the individual connections by computing the neighbourhood gradients
    connections: list[list[Any]] = [[]] * mesh.n_points
    for key, _ in enumerate(scalars):
        neighbour_values = scalars[neighbours[key]]
        differences = neighbour_values - scalars[key]
        indices = neighbours[key][np.argmax(differences)]
        connections[key] = [key, indices]
    domains = merge(connections)
    domains = [np.array(list(domain)) for domain in domains]

    # Set domain values
    output = np.zeros(mesh.n_points, int)
    for ii, domain_members in enumerate(domains):
        output[domain_members] = ii

    logger.info(f"Found {len(domains)} domains")

    return output


def relabel(domains: NDArray[np.integer[Any]], order: Sequence[Sequence[int]]) -> NDArray[np.integer[Any]]:
    """Relabel domains based on a given order.

    Args:
        domains: Array of domain labels
        order: List of domain groups to merge, where each group contains
            domain indices that should be assigned the same new label

    Returns:
        Relabeled domain array
    """

    # TODO write mapping function for this - should not be list
    output: NDArray[np.integer[Any]] = np.zeros(len(domains), int)
    for ii, domain_members in enumerate(order):
        output[np.isin(domains, domain_members)] = ii
    return output


def map_to_domains(domains: NDArray[np.integer[Any]], values: NDArray[Any]) -> NDArray[np.floating[Any]]:
    """Map scalar values to domain-labeled array.

    Assigns a single value to all vertices within each domain.

    Args:
        domains: Array of domain labels with length n_points
        values: Array of values to assign, with length matching number of
            unique domains

    Returns:
        Array of mapped values with length matching domains
    """

    doms = np.unique(domains)
    output = np.zeros(len(domains))
    for ii, domain_members in enumerate(doms):
        output[np.isin(domains, domain_members)] = values[ii]
    return output


def merge_angles(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    meristem_index: int,
    threshold: float = 20,
    method: str = "center_of_mass",
    verbose: bool = False,
) -> NDArray[np.integer[Any]]:
    """Merge domains based on angular separation from meristem.

    Merges domains that are within the angular threshold when measured from
    the meristem center.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of current domain labels
        meristem_index: Index of the meristem domain
        threshold: Angular threshold in degrees for merging domains
        method: Method for calculating domain center coordinates
        verbose: Print information about merging process

    Returns:
        Array of merged domain labels
    """

    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))
    if n_domains_initial == 1:
        return domains

    changed = True
    while changed:
        if method in ["center_of_mass", "com"]:
            centers = [extract_domain(mesh, domains, ii).center_of_mass() for ii in np.unique(domains)]
        else:
            raise RuntimeError(f'Method "{method}" not valid.')
        centers_arr = np.array(centers)

        angles = np.array(
            [
                np.arctan2(
                    centers_arr[ii][1] - centers_arr[meristem_index][1],
                    centers_arr[ii][2] - centers_arr[meristem_index][2],
                )
                * 360.0
                / (2 * np.pi)
                % 360
                for ii in range(len(centers_arr))
            ]
        )

        # reorder domains based on angles
        order = np.argsort(angles)
        new_domains = domains.copy()

        indices = order.copy()
        indices = indices[1:]  # take out meristem
        angles = angles[indices]

        diffs = np.diff(angles, prepend=angles[-1]) % 360
        hits = np.where(diffs < threshold)[0]

        to_merge: list[list[int]] = [[meristem_index]] + [[int(ii)] for ii in indices]
        for ii in hits:
            to_merge.append([int(indices[ii - 1]), int(indices[(ii) % len(indices)])])

        domain_labels = merge(to_merge)
        domain_labels_arr = np.array([np.array(list(domain_labels[ii])) for ii in range(len(domain_labels))])

        domains = relabel(new_domains, domain_labels_arr)
        meristem_index = 0

        changed = len(domain_labels_arr) < len(np.unique(new_domains))

    # Set domain values
    n_domains_new = len(np.unique(new_domains))
    logger.info(f"Merging {n_domains_initial} domains to {n_domains_new}.")

    output = domains

    return output


def merge_distance(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    threshold: float,
    scalars: NDArray[np.floating[Any]] | None = None,
    method: str = "center_of_mass",
    metric: str = "euclidean",
    verbose: bool = False,
) -> NDArray[np.integer[Any]]:
    """Merge domains based on spatial distance between domain centers.

    Uses KDTree to find domains within the distance threshold.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of current domain labels
        threshold: Distance threshold for merging domains
        scalars: Optional scalar array for distance calculation
        method: Method for calculating domain center
        metric: Distance metric ('euclidean' or 'geodesic')
        verbose: Print information about merging process

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If method or metric is invalid
    """
    method = method.lower()
    metric = metric.lower()
    n_domains_initial = len(np.unique(domains))

    # Define a method to use
    coords: list[Any]
    if method in ["center_of_mass", "com"]:
        coords = [extract_domain(mesh, domains, ii).center_of_mass() for ii in np.unique(domains)]
    elif scalars is None:
        raise RuntimeError(f'Method "{method}" not viable without valid scalar input.')
    elif method in ["maximum", "max", "minimum", "min"] and len(scalars) == mesh.n_points:
        coords = []
        fct: Callable[..., Any] | None = (
            np.max if method in ["maximum", "max"] else np.min if method in ["minimum", "min"] else None
        )
        if fct is None:
            raise RuntimeError(f'Method "{method}" not valid.')
        for ii in np.unique(domains):
            extremum = fct(scalars[domains == ii])
            index = np.where(np.logical_and(scalars == extremum, domains == ii))[0][0]
            coords.append(mesh.points[index])
    else:
        raise RuntimeError(f'Method "{method}" not valid.')
    coords_arr = np.array(coords)

    # Find BoAs within certain distance of each other according to a given metric
    groups: list[list[int]]
    if metric == "euclidean":
        tree = scipy.spatial.cKDTree(coords_arr)
        groups = tree.query_ball_point(coords_arr, threshold)
    elif metric == "geodesic":
        indices = np.array([mesh.FindPoint(pt) for pt in coords_arr])
        groups = []
        for ii, index1 in enumerate(indices):
            groups.append([ii])
            for jj, index2 in enumerate(indices):
                if mesh.geodesic_distance(index1, index2) < threshold:
                    groups[-1].append(jj)
    else:
        raise RuntimeError(f'Metric "{metric}" not valid.')
    groups_merged = merge(groups)
    groups_arr = np.array([np.array(list(ii)) for ii in groups_merged])

    # Merge domains
    logger.info(f"Merging {n_domains_initial} domains to {len(groups_arr)}.")
    output = relabel(domains, groups_arr)

    return output


def extract_domain(mesh: pv.PolyData, domains: NDArray[np.integer[Any]], index: int) -> pv.PolyData:
    """Extract a single domain as a separate mesh.

    Args:
        mesh: PyVista mesh containing all domains
        domains: Array of domain labels
        index: Domain index to extract

    Returns:
        PyVista mesh containing only the specified domain
    """

    extracted = mesh.remove_points(domains != index)[0]
    return extracted


def neighbouring_domains(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    seed: int,
    neighbours: list[NDArray[np.intp]] | None = None,
) -> NDArray[np.integer[Any]]:
    """Get indices of domains adjacent to a specified domain.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        seed: Domain index to find neighbors for
        neighbours: Optional neighbor connectivity array

    Returns:
        Array of neighboring domain indices

    Raises:
        RuntimeError: If domains array is invalid
    """

    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.vertex_neighbors_all(mesh)

    in_domain = np.where(domains == seed)[0]

    neighs_to_domain_boundary = np.unique(flatten(list(np.take(neighbours, in_domain))))
    neigh_domains = domains[neighs_to_domain_boundary][domains[neighs_to_domain_boundary] != seed]
    neigh_domains = np.unique(neigh_domains)

    return neigh_domains


def border(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    index1: int,
    index2: int,
    neighbours: list[NDArray[np.intp]] | None = None,
) -> NDArray[np.intp] | list[Any]:
    """Get vertex indices forming the border between two domains.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        index1: Index of the first domain
        index2: Index of the second domain
        neighbours: Optional neighbor connectivity array

    Returns:
        Array of vertex indices on the border between the two domains

    Raises:
        RuntimeError: If indices are invalid or domains array is invalid
    """

    if np.isin([index1, index2], domains).all():
        if index1 == index2:
            raise RuntimeError(f"index1 and index2 both {index1}")
    else:
        raise RuntimeError("Indices must be in domains.")
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.vertex_neighbors_all(mesh)

    _, in_1 = get_domain_boundary(mesh, domains, index1, return_indices=True)
    _, in_2 = get_domain_boundary(mesh, domains, index2, return_indices=True)

    if in_1.shape[0] == 0 or in_2.shape[0] == 0:
        return []

    neighs_1 = flatten(list(np.take(neighbours, in_1)))
    neighs_2 = flatten(list(np.take(neighbours, in_2)))
    border_indices = np.union1d(np.intersect1d(neighs_1, in_2), np.intersect1d(neighs_2, in_1))

    return border_indices


def merge_engulfing(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    threshold: float = 0.9,
    neighbours: list[NDArray[np.intp]] | None = None,
    verbose: bool = False,
) -> NDArray[np.floating[Any]]:
    """Merge domains that are mostly encircled by a neighboring domain.

    Merges domains where a single neighbor borders more than the threshold
    fraction of the domain boundary.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        threshold: Fraction of boundary that must be shared for merging
        neighbours: Optional neighbor connectivity array
        verbose: Print information about merging process

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If domains array is invalid
    """
    # Make checks and calculate neighbours if we don't have them.
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.vertex_neighbors_all(mesh)

    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))

    changed = True
    while changed:
        changed = False

        # For every domain, find points facing other domains and domains facing NaN
        bidxs = get_boundary_indices(mesh)
        if len(bidxs) == 0:
            break

        for domain in np.unique(domains):
            in_domain = np.where(domains == domain)[0]

            in_domain_global_boundary = np.intersect1d(in_domain, bidxs)

            neighs_to_domain_boundary = np.unique(flatten(list(np.take(neighbours, in_domain))))
            neigh_domains = domains[neighs_to_domain_boundary][domains[neighs_to_domain_boundary] != domain]
            neigh_domains, counts = np.unique(neigh_domains, return_counts=True)

            # calculate fraction of whole circumference bordering this neighbour
            border_frac = float(counts.max()) / (counts.sum() + len(in_domain_global_boundary))

            # merge if appropriate
            if border_frac > threshold:
                new_domain = neigh_domains[np.argmax(counts)]
                domains[domains == domain] = new_domain
                changed = True

    logger.info(f"Merging {n_domains_initial} domains to {len(np.unique(domains))}")
    output = np.zeros(mesh.n_points, "float")
    for new_domain, old_domain in enumerate(np.unique(domains)):
        output[domains == old_domain] = new_domain

    return output


def merge_small(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    threshold: float,
    metric: str = "points",
    mode: str = "border",
    neighbours: list[NDArray[np.intp]] | None = None,
    verbose: bool = False,
) -> NDArray[np.floating[Any]]:
    """Merge domains smaller than threshold to their largest neighbor.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        threshold: Size threshold below which domains are merged
        metric: Size metric to use for merging ('points' or 'area')
        mode: Merge strategy ('border' or 'area')
        neighbours: Optional neighbor connectivity array
        verbose: Print information about merging process

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If domains array is invalid
    """

    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.vertex_neighbors_all(mesh)

    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))

    changed = True
    while changed:
        changed = False  # new round
        d_labels, d_sizes = np.unique(domains, return_counts=True)
        to_merge: list[list[int]] = []

        if metric in ["points", "p", "point", "n_points", "npoints", "np"]:
            probes = d_labels[d_sizes < threshold]
        elif metric in ["area", "a"]:
            d_sizes = np.array([extract_domain(mesh, domains, dd).area for dd in d_labels])
            probes = d_labels[d_sizes < threshold]
            changed = len(probes) > 0
        else:
            probes = d_labels[d_sizes < threshold]

        for probe in probes:
            probe_d_neighbours = neighbouring_domains(mesh, domains, probe, neighbours=neighbours)
            if mode == "border":
                d_borders = [border(mesh, domains, probe, ii, neighbours=neighbours) for ii in probe_d_neighbours]
                d_border_sizes = [len(bb) for bb in d_borders]
                to_merge.append([probe, probe_d_neighbours[np.argmax(d_border_sizes)]])
            elif mode == "area":
                d_neighbour_areas = [extract_domain(mesh, domains, pp).area for pp in probe_d_neighbours]
                to_merge.append([probe, probe_d_neighbours[np.argmax(d_neighbour_areas)]])

        if changed:
            doms = merge(to_merge)
            domains_overwrite = domains.copy()
            for ii in range(len(doms)):
                domains_overwrite[np.isin(domains, list(doms[ii]))] = ii

            domains = domains_overwrite

    logger.info(f"Merging {n_domains_initial} domains to {len(np.unique(domains))}.")
    output = np.zeros(mesh.n_points, "float")
    for new_domain, old_domain in enumerate(np.unique(domains)):
        output[domains == old_domain] = new_domain

    return output


def merge_disconnected(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    meristem_index: int,
    threshold: float | None,
    neighbours: list[NDArray[np.intp]] | None = None,
    verbose: bool = False,
) -> NDArray[np.floating[Any]]:
    """Merge domains disconnected from meristem to nearest connected domain.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        meristem_index: Index of the meristem domain
        threshold: Unused parameter (kept for API compatibility)
        neighbours: Optional neighbor connectivity array
        verbose: Print information about merging process

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If domains array is invalid
    """

    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mp.vertex_neighbors_all(mesh)
    if threshold is None:
        return domains.astype(float)

    meristem_idx = int(meristem_index)
    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))

    _, meristem_boundary = get_domain_boundary(mesh, domains, meristem_idx, return_indices=True)

    changed = True
    while changed:
        changed = False  # new round
        d_labels = np.unique(domains)

        # Get all borders to meristem. Figure out which are disconnected
        borders = [
            border(mesh, domains, meristem_idx, ii, neighbours=neighbours) for ii in d_labels[d_labels != meristem_idx]
        ]
        mask = np.array([len(borders[ii]) for ii in range(len(borders))]) == 0
        to_merge: list[list[int]] = [[meristem_idx]] + [
            [int(ii)] for ii in d_labels[d_labels != meristem_idx][np.logical_not(mask)]
        ]
        probes = d_labels[d_labels != meristem_idx][mask]
        meristem_idx = 0

        # Merge with neighbours with most vertices in the corresponding border
        for probe in np.sort(probes):
            probe_borders = [
                border(mesh, domains, probe, jj, neighbours=neighbours) for jj in d_labels[d_labels != probe]
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

    logger.info(f"Merging {n_domains_initial} domains to {len(np.unique(domains))}.")
    output = np.zeros(mesh.n_points, "float")
    for new_domain, old_domain in enumerate(np.unique(domains)):
        output[domains == old_domain] = new_domain

    return output


def merge_depth(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    scalars: NDArray[np.floating[Any]],
    threshold: float = 0.0,
    neighbours: list[NDArray[np.intp]] | None = None,
    exclude_boundary: bool = False,
    min_points: int = 0,
    mode: str = "max",
    verbose: bool = False,
) -> NDArray[np.integer[Any]]:
    """Merge domains based on depth similarity in scalar field.

    Merges neighboring domains if their depth values differ by less than threshold.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        scalars: Scalar array representing depth or curvature values
        threshold: Maximum depth difference for merging
        neighbours: Optional neighbor connectivity array
        exclude_boundary: Exclude boundary vertices from depth calculation
        min_points: Minimum number of border points required for merging
        mode: Aggregation mode for domain depth ('min', 'max', 'median', 'mean')
        verbose: Print information about merging process

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If domains or scalars arrays are invalid
    """

    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domain array.")
    if len(scalars) != mesh.n_points or scalars.ndim > 1:
        raise RuntimeError("Invalid scalar array.")
    if neighbours is None:
        neighbours = mp.vertex_neighbors_all(mesh)

    domains = domains.copy()
    boundary = get_boundary_indices(mesh)
    n_domains_initial = np.unique(domains).shape[0]

    fct_map: dict[str, Callable[..., Any]] = {"min": np.min, "median": np.median, "max": np.max}
    fct: Callable[..., Any] = fct_map.get(mode, np.mean)

    changed = True
    while changed:
        changed = False
        to_merge: list[list[int]] = []

        for dom in np.unique(domains):
            in_domain = np.where(domains == dom)[0]
            max_value: float = float(np.max(scalars[in_domain]))

            # get the points that are in neighbouring domains
            neighs_pts: list[Any] = [x for y in [neighbours[i] for i in in_domain] for x in y]
            neighs_pts = [x for x in neighs_pts if x not in in_domain]

            in_domain_list: list[int]
            if exclude_boundary:
                in_domain_list = [x for x in in_domain if x not in boundary]
                neighs_pts = [x for x in neighs_pts if x not in boundary]
            else:
                in_domain_list = list(in_domain)

            # neighbouring domains, in order
            neighs_doms = np.unique(domains[neighs_pts])
            neighs_doms = np.sort(neighs_doms)

            for neigh_dom in neighs_doms:
                # all the points in the neighbouring domain which has a neighbour in
                # the current domain
                border_pts = np.where(domains == neigh_dom)[0]
                border_pts_list: list[Any] = [x for x in border_pts if x in neighs_pts]

                # get neighbours of the neighbour's neighbours that are in the current
                # domain. Merge.
                nested_neighs = [neighbours[i] for i in border_pts_list]
                border_pts_neighs = np.unique(np.array([x for y in nested_neighs for x in y]))
                pts_to_add = [pt for pt in border_pts_neighs if pt in in_domain_list]
                border_pts_arr: NDArray[Any] = np.append(border_pts_list, pts_to_add)
                border_pts_arr = np.unique(border_pts_arr)

                # Only do if enough border
                if len(border_pts_arr) < min_points:
                    continue

                # Merge
                border_max_value = fct(scalars[border_pts_arr])
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
    logger.info(f"Merging {n_domains_initial} domains to {len(np.unique(domains))}.")
    new_domains = domains.copy()
    for ii, domain in enumerate(np.unique(domains)):
        new_domains[domains == domain] = ii

    return new_domains


def merge_borders_by_length(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    threshold: float = 0.0,
    neighbours: list[NDArray[np.intp]] | None = None,
    verbose: bool = False,
) -> NDArray[np.integer[Any]]:
    """Merge domains based on shared border length.

    Merges domains that share a border longer than the threshold.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        threshold: Minimum border length for merging
        neighbours: Optional neighbor connectivity array
        verbose: Print information about merging process

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If domains array is invalid
    """

    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domain array.")
    if neighbours is None:
        neighbours = mp.vertex_neighbors_all(mesh)

    domains = domains.copy()
    n_domains_initial = np.unique(domains).shape[0]

    changed = True
    while changed:
        changed = False
        to_merge: list[list[int]] = []

        for dom in np.unique(domains):
            in_domain = np.where(domains == dom)[0]

            # get the points that are in neighbouring domains
            neighs_pts: list[Any] = [x for y in [neighbours[i] for i in in_domain] for x in y]
            neighs_pts = [x for x in neighs_pts if x not in in_domain]

            # neighbouring domains, in order
            neighs_doms = np.unique(domains[neighs_pts])
            neighs_doms = np.sort(neighs_doms)

            for neigh_dom in neighs_doms:
                # all the points in the neighbouring domain which has a neighbour in
                # the current domain
                border_pts = np.where(domains == neigh_dom)[0]
                border_pts_list: list[Any] = [x for x in border_pts if x in neighs_pts]

                # get neighbours of the neighbour's neighbours that are in the current
                # domain. Merge.
                nested_neighs = [neighbours[i] for i in border_pts_list]
                border_pts_neighs = np.unique(np.array([x for y in nested_neighs for x in y]))
                pts_to_add = [pt for pt in border_pts_neighs if pt in in_domain]
                border_pts_arr: NDArray[Any] = np.append(border_pts_list, pts_to_add)
                border_pts_arr = np.unique(border_pts_arr)

                # Merge
                if len(border_pts_arr) > threshold:
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
    logger.info(f"Merging {n_domains_initial} domains to {len(np.unique(domains))}.")
    new_domains = domains.copy()
    for ii, domain in enumerate(np.unique(domains)):
        new_domains[domains == domain] = ii

    return new_domains


def get_boundary_indices(mesh: pv.PolyData) -> NDArray[np.intp]:
    """Get vertex indices of mesh boundary points.

    Args:
        mesh: PyVista mesh

    Returns:
        Sorted array of boundary vertex indices
    """

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
    indices = [np.where(np.all(fepts[ii] == pts, axis=1))[0][0] for ii in range(len(fepts))]
    indices_arr = np.sort(indices)

    return indices_arr


def set_boundary_values(
    mesh: pv.PolyData,
    scalars: NDArray[np.floating[Any]],
    values: float,
) -> NDArray[np.floating[Any]]:
    """Set scalar values for all boundary vertices.

    Args:
        mesh: PyVista mesh
        scalars: Scalar array to modify
        values: Value to assign to boundary vertices

    Returns:
        Modified scalar array with boundary values set
    """

    new_scalars = scalars.copy()
    boundary = get_boundary_indices(mesh)
    if len(boundary) > 0:
        new_scalars[boundary] = values
    return new_scalars


def filter_curvature(
    curvs: NDArray[np.floating[Any]],
    neighs: list[NDArray[np.intp]] | None,
    fct: Callable[..., Any],
    iters: int,
    exclude: list[int] | None = None,
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """Filter curvature values using neighborhood aggregation.

    Args:
        curvs: Curvature array to filter
        neighs: Neighbor connectivity array for each vertex
        fct: Aggregation function (e.g., np.mean, np.median)
        iters: Number of filter iterations
        exclude: List of vertex indices to exclude from filtering
        **kwargs: Additional arguments (unused)

    Returns:
        Filtered curvature array
    """

    if exclude is None:
        exclude = []
    if neighs is None:
        return curvs
    for _ii in range(iters):
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


def mean(
    scalars: NDArray[np.floating[Any]],
    neighs: list[NDArray[np.intp]],
    iters: int,
    exclude: list[int] | None = None,
) -> NDArray[np.floating[Any]]:
    """Apply mean filter to scalar array.

    Args:
        scalars: Scalar array to filter
        neighs: Neighbor connectivity array
        iters: Number of filter iterations
        exclude: Vertex indices to exclude from filtering

    Returns:
        Filtered scalar array
    """
    if exclude is None:
        exclude = []
    return filter_curvature(scalars, neighs, np.mean, iters, exclude)


def filter_scalars(
    scalars: NDArray[np.floating[Any]],
    neighs: list[NDArray[np.intp]],
    fct: Callable[..., Any],
    iters: int,
    exclude: list[int] | None = None,
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """Filter scalar values using neighborhood aggregation.

    Args:
        scalars: Scalar array to filter
        neighs: Neighbor connectivity array for each vertex
        fct: Aggregation function (e.g., np.mean, np.median, np.min, np.max)
        iters: Number of filter iterations
        exclude: List of vertex indices to exclude from filtering
        **kwargs: Additional arguments (unused)

    Returns:
        Filtered scalar array
    """

    if exclude is None:
        exclude = []
    for _ii in range(iters):
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


def remove_size(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    threshold: float,
    method: str = "points",
    relative: str = "largest",
) -> pv.PolyData:
    """Remove domains smaller than threshold from mesh.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        threshold: Size threshold for removal
        method: Size metric ('points' for vertex count, 'area' for surface area)
        relative: Reference for threshold ('largest', 'all', or 'absolute')

    Returns:
        PyVista mesh with small domains removed

    Raises:
        RuntimeError: If method or relative parameter is invalid
    """

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
    reference: float
    if relative == "all":
        reference = float(np.sum(sizes))
    elif relative == "largest":
        reference = float(np.max(sizes))
    elif relative in ["absolute", "abs"]:
        reference = 1.0
    else:
        raise RuntimeError("Invalid comparison option.")

    to_remove = np.where(sizes / reference < threshold)[0]
    to_remove = groups[to_remove]
    to_remove_mask = np.isin(domains, to_remove)

    output = mesh.remove_points(to_remove_mask, keep_scalars=True)[0]

    return output


def get_domain(mesh: pv.PolyData, pdata: pd.DataFrame, domain: int, **kwargs: Any) -> pv.PolyData:
    """Extract a single domain from labeled mesh using point data.

    Args:
        mesh: PyVista mesh containing all domains
        pdata: DataFrame with domain labels in 'domain' column
        domain: Domain index to extract
        **kwargs: Additional arguments (unused)

    Returns:
        PyVista mesh containing only the specified domain
    """

    not_in_domain = pdata.loc[pdata.domain != domain].index.values
    mask = np.zeros((mesh.points.shape[0],), dtype=bool)
    mask[not_in_domain] = True
    return pv.PolyData(mesh.remove_points(mask)[0])


def get_domain_boundary(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    index: int,
    return_indices: bool = False,
) -> pv.PolyData | tuple[pv.PolyData, NDArray[np.intp]]:
    """Get boundary edges and optionally vertex indices for a domain.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        index: Domain index to extract boundary for
        return_indices: If True, return vertex indices in addition to edges

    Returns:
        Boundary edges mesh, and optionally array of boundary vertex indices
    """

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


def domain_neighbors(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    neighs: list[NDArray[np.intp]],
) -> list[int]:
    """Count number of neighboring domains for each domain.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        neighs: Neighbor connectivity array for each vertex

    Returns:
        List of neighbor counts, one per unique domain
    """

    doms = [extract_domain(mesh, domains, dd) for dd in np.unique(domains)]
    dom_boundaries = [get_boundary_indices(dd) for dd in doms]
    doms_orig_indices: list[list[int]] = []
    for ii, dom in enumerate(doms):
        orig = [mesh.FindPoint(pt) for pt in dom.points[dom_boundaries[ii]]]
        doms_orig_indices.append(orig)

    neighs_arr = np.array(neighs.copy(), dtype=object)
    n_neighs: list[int] = []
    for dom_orig_indices in doms_orig_indices:
        dom_neighs = flatten([list(domains[dd]) for dd in neighs_arr[dom_orig_indices]])
        dom_neighs_unique = np.unique(dom_neighs)
        n_neighs.append(len(dom_neighs_unique) - 1)
    return n_neighs


def define_meristem(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    method: str = "center_of_mass",
    return_coordinates: bool = False,
    neighs: list[NDArray[np.intp]] | None = None,
) -> int | tuple[int, NDArray[np.floating[Any]]]:
    """Identify which domain corresponds to the meristem.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        method: Method for meristem identification
        return_coordinates: If True, return meristem center coordinates
        neighs: Optional neighbor connectivity array

    Returns:
        Meristem domain index, and optionally its center coordinates
    """

    method = method.lower()
    coord: NDArray[np.floating[Any]]

    if method in ["center_of_mass", "com"]:
        coord = np.array(mesh.center_of_mass())
    elif method in ["center", "c", "bounds"]:
        coord = np.mean(np.reshape(mesh.bounds, (3, -1)), axis=1)
    elif method in ["n_neighs", "neighbors", "neighs", "n_neighbors"]:
        if neighs is None:
            neighs = mp.vertex_neighbors_all(mesh)
        doms = np.unique(domains)
        n_neighs = domain_neighbors(mesh, domains, neighs)
        meristem_dom = doms[np.argmax(n_neighs)]
        coord = np.array(extract_domain(mesh, domains, meristem_dom).center_of_mass())
    else:
        coord = np.array(mesh.center_of_mass())

    meristem = int(domains[mesh.FindPoint(coord)])

    if return_coordinates:
        return meristem, coord
    else:
        return meristem


def extract_domaindata(
    pdata: pd.DataFrame,
    mesh: pv.PolyData,
    apex: NDArray[np.floating[Any]],
    meristem: int,
    **kwargs: Any,
) -> pd.DataFrame:
    """Extract geometric and spatial data for each domain.

    Computes distance to boundary, distance to center of mass, angle from apex,
    surface area, maximum diameter, and meristem flag for each domain.

    Args:
        pdata: DataFrame with domain labels in 'domain' column
        mesh: PyVista mesh containing the domains
        apex: Apex coordinates as 3-element array
        meristem: Meristem domain index
        **kwargs: Additional arguments (unused)

    Returns:
        DataFrame with domain measurements and properties
    """
    domains_arr = np.unique(pdata.domain)
    domains_arr = domains_arr[~np.isnan(domains_arr)]
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
        dtype=object,
    )

    for ii in domains_arr:
        # Get distance for closest boundary point to apex
        dom = get_domain(mesh, pdata, int(ii))
        dom_boundary = get_boundary_indices(dom)
        dom_boundary_coords = dom.points[dom_boundary]
        dom_boundary_dists = np.sqrt(np.sum((dom_boundary_coords - apex) ** 2, axis=1))
        d2boundary: float = float(np.min(dom_boundary_dists))

        # Get distance for center of mass from apex
        center = np.array(dom.center_of_mass())
        d2com = float(np.sqrt(np.sum((center - apex) ** 2)))

        # Get domain size in terms of bounding boxes
        domcoords = dom.points

        dists = cdist(domcoords, domcoords)
        maxdiam: float = float(np.max(dists))
        dists_xy = cdist(domcoords[:, 1:], domcoords[:, 1:])
        maxdiam_xy: float = float(np.max(dists_xy))

        # Get domain angle in relation to apex
        rel_pos = center - apex
        angle_val = float(np.arctan2(rel_pos[1], rel_pos[2]))  # angle in yz-plane
        if angle_val < 0:
            angle_val += 2.0 * np.pi
        angle_val *= 360 / (2.0 * np.pi)

        # Get surface area
        area = dom.area

        # Define type
        ismeristem = ii == meristem
        if ismeristem:
            angle_val = np.nan

        # Set data
        ddata.loc[int(ii)] = [
            int(ii),
            d2boundary,
            d2com,
            angle_val,
            area,
            maxdiam,
            maxdiam_xy,
            tuple(center),
            ismeristem,
        ]
    ddata = ddata.infer_objects()
    ddata = ddata.sort_values(["ismeristem", "area"], ascending=False)
    return ddata


def relabel_domains(
    pdata: pd.DataFrame,
    ddata: pd.DataFrame,
    order: str = "area",
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel domains based on sorting criteria.

    Reorders domain labels based on size metrics, keeping meristem as label 0.

    Args:
        pdata: DataFrame with domain labels to update
        ddata: DataFrame with domain measurements
        order: Sorting metric ('area', 'maxdiam', 'maxdiam_xy')
        **kwargs: Additional arguments (unused)

    Returns:
        Tuple of (updated pdata, updated ddata) with relabeled domains
    """
    new_pdata = pdata.copy()
    new_ddata = ddata.copy()

    if order == "area":
        new_ddata = new_ddata.sort_values(["ismeristem", "area"], ascending=False)
    elif order == "maxdiam":
        new_ddata = new_ddata.sort_values(["ismeristem", "maxdiam"], ascending=False)
    elif order == "maxdiam_xy":
        new_ddata = new_ddata.sort_values(["ismeristem", "maxdiam_xy"], ascending=False)

    dmap: dict[int, int] = {}
    for ii in range(len(new_ddata)):
        old_dom = new_ddata.iloc[ii].domain
        dmap[old_dom] = ii
        new_ddata["domain"].iloc[ii] = ii

    for ii in dmap:
        new_pdata.loc[pdata.domain == ii, "domain"] = dmap[ii]

    return new_pdata, new_ddata
