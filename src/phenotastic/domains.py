import copy
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
import pyvista as pv
import scipy
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

import phenotastic.mesh as mesh_ops
from phenotastic.misc import flatten, merge


def median(
    scalars: NDArray[np.floating[Any]],
    neighbors: list[NDArray[np.intp]] | None = None,
    iterations: int = 1,
) -> NDArray[np.floating[Any]]:
    """Apply median filter to mesh-based scalar arrays.

    Args:
        scalars: Scalar array associated with mesh vertices
        neighbors: Neighbor connectivity array for each vertex
        iterations: Number of filter iterations to apply

    Returns:
        Filtered scalar array
    """

    scalars = scalars.copy()
    for _ii in range(iterations):
        scalars = filter_curvature(scalars, neighbors, np.median, 1)
    return scalars


def minmax(
    scalars: NDArray[np.floating[Any]],
    neighbors: list[NDArray[np.intp]] | None = None,
    iterations: int = 1,
) -> NDArray[np.floating[Any]]:
    """Apply min-max filter to mesh-based scalar arrays.

    Args:
        scalars: Scalar array associated with mesh vertices
        neighbors: Neighbor connectivity array for each vertex
        iterations: Number of filter iterations to apply

    Returns:
        Filtered scalar array
    """

    scalars = scalars.copy()
    for _ii in range(iterations):
        scalars = filter_curvature(scalars, neighbors, np.min, 1)
        scalars = filter_curvature(scalars, neighbors, np.max, 1)
    return scalars


def maxmin(
    scalars: NDArray[np.floating[Any]],
    neighbors: list[NDArray[np.intp]] | None = None,
    iterations: int = 1,
) -> NDArray[np.floating[Any]]:
    """Apply max-min filter to mesh-based scalar arrays.

    Args:
        scalars: Scalar array associated with mesh vertices
        neighbors: Neighbor connectivity array for each vertex
        iterations: Number of filter iterations to apply

    Returns:
        Filtered scalar array
    """

    scalars = scalars.copy()
    for _ii in range(iterations):
        scalars = filter_curvature(scalars, neighbors, np.max, 1)
        scalars = filter_curvature(scalars, neighbors, np.min, 1)
    return scalars


def steepest_ascent(
    mesh: pv.PolyData,
    scalars: NDArray[np.floating[Any]],
    neighbours: list[NDArray[np.intp]] | None = None,
) -> NDArray[np.integer[Any]]:
    """Create domains using steepest ascent approach.

    Connects vertices based on the steepest local gradient in the scalar field.

    Args:
        mesh: PyVista mesh to create domains for
        scalars: 1D scalar array with length matching mesh.n_points
        neighbours: Neighbor connectivity array for each vertex

    Returns:
        Array of domain labels with length mesh.n_points

    Raises:
        RuntimeError: If scalar array has invalid dimensions
    """

    # Make checks and calculate neighbours if we don't have them.
    if (len(scalars) != mesh.n_points) or scalars.ndim > 1:
        raise RuntimeError("Invalid scalar array.")
    if neighbours is None:
        neighbours = mesh_ops.get_vertex_neighbors_all(mesh)

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

    unique_domains = np.unique(domains)
    output = np.zeros(len(domains))
    for ii, domain_members in enumerate(unique_domains):
        output[np.isin(domains, domain_members)] = values[ii]
    return output


def merge_angles(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    meristem_index: int,
    threshold: float = 20,
    method: str = "center_of_mass",
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
            ],
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
        domain_labels_list: list[list[int]] = [list(domain_labels[ii]) for ii in range(len(domain_labels))]

        domains = relabel(new_domains, domain_labels_list)
        meristem_index = 0

        changed = len(domain_labels_list) < len(np.unique(new_domains))

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
        aggregation_func: Callable[..., Any] | None = (
            np.max if method in ["maximum", "max"] else np.min if method in ["minimum", "min"] else None
        )
        if aggregation_func is None:
            raise RuntimeError(f'Method "{method}" not valid.')
        for ii in np.unique(domains):
            extremum = aggregation_func(scalars[domains == ii])
            index = np.where(np.logical_and(scalars == extremum, domains == ii))[0][0]
            coords.append(mesh.points[index])
    else:
        raise RuntimeError(f'Method "{method}" not valid.')
    coords_arr = np.array(coords)

    # Find BoAs within certain distance of each other according to a given metric
    groups: list[list[int]]
    if metric == "euclidean":
        tree = scipy.spatial.cKDTree(coords_arr)
        groups = tree.query_ball_point(coords_arr, threshold)  # type: ignore[invalid-assignment]  # scipy stubs
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
    groups_list: list[list[int]] = [list(ii) for ii in groups_merged]

    # Merge domains
    logger.info(f"Merging {n_domains_initial} domains to {len(groups_list)}.")
    output = relabel(domains, groups_list)

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

    extracted: pv.PolyData = mesh.remove_points(domains != index)[0]
    return extracted


def get_neighbouring_domains(
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
        neighbours = mesh_ops.get_vertex_neighbors_all(mesh)

    in_domain = np.where(domains == seed)[0]

    neighs_to_domain_boundary = np.unique(flatten(list(np.take(neighbours, in_domain))))
    neigh_domains = domains[neighs_to_domain_boundary][domains[neighs_to_domain_boundary] != seed]
    neigh_domains = np.unique(neigh_domains)

    return neigh_domains


def compute_border(
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
        neighbours = mesh_ops.get_vertex_neighbors_all(mesh)

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
) -> NDArray[np.integer[Any]]:
    """Merge domains that are mostly encircled by a neighboring domain.

    Merges domains where a single neighbor borders more than the threshold
    fraction of the domain boundary.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        threshold: Fraction of boundary that must be shared for merging
        neighbours: Optional neighbor connectivity array

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If domains array is invalid
    """
    # Make checks and calculate neighbours if we don't have them.
    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mesh_ops.get_vertex_neighbors_all(mesh)

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
    output = np.zeros(mesh.n_points, dtype=np.int64)
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
) -> NDArray[np.integer[Any]]:
    """Merge domains smaller than threshold to their largest neighbor.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        threshold: Size threshold below which domains are merged
        metric: Size metric to use for merging ('points' or 'area')
        mode: Merge strategy ('border' or 'area')
        neighbours: Optional neighbor connectivity array

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If domains array is invalid
    """

    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mesh_ops.get_vertex_neighbors_all(mesh)

    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))

    changed = True
    while changed:
        changed = False  # new round
        domain_labels, domain_sizes = np.unique(domains, return_counts=True)
        to_merge: list[list[int]] = []

        if metric in ["points", "p", "point", "n_points", "npoints", "np"]:
            probes = domain_labels[domain_sizes < threshold]
        elif metric in ["area", "a"]:
            domain_sizes = np.array([extract_domain(mesh, domains, domain_id).area for domain_id in domain_labels])
            probes = domain_labels[domain_sizes < threshold]
            changed = len(probes) > 0
        else:
            probes = domain_labels[domain_sizes < threshold]

        for probe in probes:
            probe_d_neighbours = get_neighbouring_domains(mesh, domains, probe, neighbours=neighbours)
            if mode == "border":
                d_borders = [
                    compute_border(mesh, domains, probe, ii, neighbours=neighbours) for ii in probe_d_neighbours
                ]
                d_border_sizes = [len(bb) for bb in d_borders]
                to_merge.append([probe, probe_d_neighbours[np.argmax(d_border_sizes)]])
            elif mode == "area":
                d_neighbour_areas = [extract_domain(mesh, domains, pp).area for pp in probe_d_neighbours]
                to_merge.append([probe, probe_d_neighbours[np.argmax(d_neighbour_areas)]])

        if changed:
            merged_groups = merge(to_merge)
            domains_overwrite = domains.copy()
            for ii in range(len(merged_groups)):
                domains_overwrite[np.isin(domains, list(merged_groups[ii]))] = ii

            domains = domains_overwrite

    logger.info(f"Merging {n_domains_initial} domains to {len(np.unique(domains))}.")
    output = np.zeros(mesh.n_points, dtype=np.int64)
    for new_domain, old_domain in enumerate(np.unique(domains)):
        output[domains == old_domain] = new_domain

    return output


def merge_disconnected(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    meristem_index: int,
    threshold: float | None,
    neighbours: list[NDArray[np.intp]] | None = None,
) -> NDArray[np.integer[Any]]:
    """Merge domains disconnected from meristem to nearest connected domain.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        meristem_index: Index of the meristem domain
        threshold: Distance threshold; if None, returns domains unchanged
        neighbours: Optional neighbor connectivity array

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If domains array is invalid
    """

    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domains array.")
    if neighbours is None:
        neighbours = mesh_ops.get_vertex_neighbors_all(mesh)
    if threshold is None:
        return domains.astype(np.int64)

    meristem_idx = int(meristem_index)
    domains = domains.copy()
    n_domains_initial = len(np.unique(domains))

    changed = True
    while changed:
        changed = False  # new round
        domain_labels = np.unique(domains)

        # Get all borders to meristem. Figure out which are disconnected
        borders = [
            compute_border(mesh, domains, meristem_idx, ii, neighbours=neighbours)
            for ii in domain_labels[domain_labels != meristem_idx]
        ]
        mask = np.array([len(borders[ii]) for ii in range(len(borders))]) == 0
        to_merge: list[list[int]] = [[meristem_idx]] + [
            [int(ii)] for ii in domain_labels[domain_labels != meristem_idx][np.logical_not(mask)]
        ]
        probes = domain_labels[domain_labels != meristem_idx][mask]
        meristem_idx = 0

        # Merge with neighbours with most vertices in the corresponding border
        for probe in np.sort(probes):
            probe_borders = [
                compute_border(mesh, domains, probe, jj, neighbours=neighbours)
                for jj in domain_labels[domain_labels != probe]
            ]
            border_sizes = [len(jj) for jj in probe_borders]
            connected_neighbour = domain_labels[domain_labels != probe][np.argmax(border_sizes)]

            to_merge.append([probe, connected_neighbour])
            changed = True

        if changed:
            merged_groups = merge(to_merge)
            domains_overwrite = domains.copy()
            for ii in range(len(merged_groups)):
                domains_overwrite[np.isin(domains, list(merged_groups[ii]))] = ii

            domains = domains_overwrite

    logger.info(f"Merging {n_domains_initial} domains to {len(np.unique(domains))}.")
    output = np.zeros(mesh.n_points, dtype=np.int64)
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
    mode: Literal["min", "max", "median", "mean"] = "max",
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
        neighbours = mesh_ops.get_vertex_neighbors_all(mesh)

    domains = domains.copy()
    boundary = get_boundary_indices(mesh)
    n_domains_initial = np.unique(domains).shape[0]

    aggregation_map: dict[str, Callable[..., Any]] = {"min": np.min, "median": np.median, "max": np.max}
    aggregation_func: Callable[..., Any] = aggregation_map.get(mode, np.mean)

    changed = True
    while changed:
        changed = False
        to_merge: list[list[int]] = []

        for current_domain in np.unique(domains):
            in_domain = np.where(domains == current_domain)[0]
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
            neighbor_domains = np.unique(domains[neighs_pts])
            neighbor_domains = np.sort(neighbor_domains)

            for neighbor_domain in neighbor_domains:
                # all the points in the neighbouring domain which has a neighbour in
                # the current domain
                border_pts = np.where(domains == neighbor_domain)[0]
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
                border_max_value = aggregation_func(scalars[border_pts_arr])
                if max_value - border_max_value < threshold:
                    to_merge.append([current_domain, neighbor_domain])
                    changed = True
                else:
                    to_merge.append([current_domain])
                    to_merge.append([neighbor_domain])

        # Update domains
        merged_groups = merge(to_merge)
        domains_overwrite = domains.copy()
        for ii, merged_domain in enumerate(merged_groups):
            domains_overwrite[np.isin(domains, list(merged_domain))] = ii
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
) -> NDArray[np.integer[Any]]:
    """Merge domains based on shared border length.

    Merges domains that share a border longer than the threshold.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        threshold: Minimum border length for merging
        neighbours: Optional neighbor connectivity array

    Returns:
        Array of merged domain labels

    Raises:
        RuntimeError: If domains array is invalid
    """

    if len(domains) != mesh.n_points or domains.ndim > 1:
        raise RuntimeError("Invalid domain array.")
    if neighbours is None:
        neighbours = mesh_ops.get_vertex_neighbors_all(mesh)

    domains = domains.copy()
    n_domains_initial = np.unique(domains).shape[0]

    changed = True
    while changed:
        changed = False
        to_merge: list[list[int]] = []

        for current_domain in np.unique(domains):
            in_domain = np.where(domains == current_domain)[0]

            # get the points that are in neighbouring domains
            neighs_pts: list[Any] = [x for y in [neighbours[i] for i in in_domain] for x in y]
            neighs_pts = [x for x in neighs_pts if x not in in_domain]

            # neighbouring domains, in order
            neighbor_domains = np.unique(domains[neighs_pts])
            neighbor_domains = np.sort(neighbor_domains)

            for neighbor_domain in neighbor_domains:
                # all the points in the neighbouring domain which has a neighbour in
                # the current domain
                border_pts = np.where(domains == neighbor_domain)[0]
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
                    to_merge.append([current_domain, neighbor_domain])
                    changed = True
                else:
                    to_merge.append([current_domain])
                    to_merge.append([neighbor_domain])

        # Update domains
        merged_groups = merge(to_merge)
        domains_overwrite = domains.copy()
        for ii, merged_domain in enumerate(merged_groups):
            domains_overwrite[np.isin(domains, list(merged_domain))] = ii
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
    curvatures: NDArray[np.floating[Any]],
    neighbors: list[NDArray[np.intp]] | None,
    aggregation_func: Callable[..., Any],
    iterations: int,
    exclude: list[int] | None = None,
) -> NDArray[np.floating[Any]]:
    """Filter curvature values using neighborhood aggregation.

    Args:
        curvatures: Curvature array to filter
        neighbors: Neighbor connectivity array for each vertex
        aggregation_func: Aggregation function (e.g., np.mean, np.median)
        iterations: Number of filter iterations
        exclude: List of vertex indices to exclude from filtering

    Returns:
        Filtered curvature array
    """

    if exclude is None:
        exclude = []
    if neighbors is None:
        return curvatures
    for _ii in range(iterations):
        new_curvatures = copy.deepcopy(curvatures)
        for jj in range(len(curvatures)):
            value = np.nan
            to_process = curvatures[[kk for kk in neighbors[jj] if kk not in exclude]]
            if len(to_process) > 0:
                value = aggregation_func(to_process)
            if not np.isnan(value):
                new_curvatures[jj] = value
        curvatures = new_curvatures
    return curvatures


def mean(
    scalars: NDArray[np.floating[Any]],
    neighbors: list[NDArray[np.intp]],
    iterations: int = 1,
    exclude: list[int] | None = None,
) -> NDArray[np.floating[Any]]:
    """Apply mean filter to scalar array.

    Args:
        scalars: Scalar array to filter
        neighbors: Neighbor connectivity array
        iterations: Number of filter iterations
        exclude: Vertex indices to exclude from filtering

    Returns:
        Filtered scalar array
    """
    return filter_curvature(scalars, neighbors, np.mean, iterations, [] if exclude is None else exclude)


def filter_scalars(
    scalars: NDArray[np.floating[Any]],
    neighbors: list[NDArray[np.intp]],
    aggregation_func: Callable[..., Any],
    iterations: int,
    exclude: list[int] | None = None,
) -> NDArray[np.floating[Any]]:
    """Filter scalar values using neighborhood aggregation.

    Args:
        scalars: Scalar array to filter
        neighbors: Neighbor connectivity array for each vertex
        aggregation_func: Aggregation function (e.g., np.mean, np.median, np.min, np.max)
        iterations: Number of filter iterations
        exclude: List of vertex indices to exclude from filtering

    Returns:
        Filtered scalar array
    """

    if exclude is None:
        exclude = []
    for _ii in range(iterations):
        new_scalars = copy.deepcopy(scalars)
        for jj in range(len(scalars)):
            value = np.nan
            to_process = scalars[[kk for kk in neighbors[jj] if kk not in exclude]]
            if len(to_process) > 0:
                value = aggregation_func(to_process)
            if not np.isnan(value):
                new_scalars[jj] = value
        scalars = new_scalars
    return scalars


def remove_small_domains(
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
        sizes = np.array([extract_domain(mesh, domains, domain_id).area for domain_id in groups])
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

    output: pv.PolyData = mesh.remove_points(to_remove_mask, keep_scalars=True)[0]

    return output


def get_domain(mesh: pv.PolyData, point_data: pd.DataFrame, domain: int) -> pv.PolyData:
    """Extract a single domain from labeled mesh using point data.

    Args:
        mesh: PyVista mesh containing all domains
        point_data: DataFrame with domain labels in 'domain' column
        domain: Domain index to extract

    Returns:
        PyVista mesh containing only the specified domain
    """

    not_in_domain = point_data.loc[point_data.domain != domain].index.values
    mask = np.zeros((mesh.points.shape[0],), dtype=bool)
    mask[not_in_domain] = True
    return pv.PolyData(mesh.remove_points(mask)[0])


@overload
def get_domain_boundary(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    index: int,
    return_indices: Literal[True],
) -> tuple[pv.PolyData, NDArray[np.intp]]: ...


@overload
def get_domain_boundary(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    index: int,
    return_indices: Literal[False] = ...,
) -> pv.PolyData: ...


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
    return edges


def count_domain_neighbors(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    neighbors: list[NDArray[np.intp]],
) -> list[int]:
    """Count number of neighboring domains for each domain.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        neighbors: Neighbor connectivity array for each vertex

    Returns:
        List of neighbor counts, one per unique domain
    """

    unique_domains = [extract_domain(mesh, domains, domain_id) for domain_id in np.unique(domains)]
    domain_boundary_indices = [get_boundary_indices(domain_mesh) for domain_mesh in unique_domains]
    domains_orig_indices: list[list[int]] = []
    for ii, domain_mesh in enumerate(unique_domains):
        orig = [mesh.FindPoint(pt) for pt in domain_mesh.points[domain_boundary_indices[ii]]]
        domains_orig_indices.append(orig)

    neighbors_array = np.array(neighbors.copy(), dtype=object)
    neighbor_counts: list[int] = []
    for domain_orig_indices in domains_orig_indices:
        domain_neighbors = flatten([list(domains[idx]) for idx in neighbors_array[domain_orig_indices]])
        domain_neighbors_unique = np.unique(domain_neighbors)
        neighbor_counts.append(len(domain_neighbors_unique) - 1)
    return neighbor_counts


def define_meristem(
    mesh: pv.PolyData,
    domains: NDArray[np.integer[Any]],
    method: str = "center_of_mass",
    return_coordinates: bool = False,
    neighbors: list[NDArray[np.intp]] | None = None,
) -> int | tuple[int, NDArray[np.floating[Any]]]:
    """Identify which domain corresponds to the meristem.

    Args:
        mesh: PyVista mesh containing the domains
        domains: Array of domain labels
        method: Method for meristem identification
        return_coordinates: If True, return meristem center coordinates
        neighbors: Optional neighbor connectivity array

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
        if neighbors is None:
            neighbors = mesh_ops.get_vertex_neighbors_all(mesh)
        neighbor_counts = count_domain_neighbors(mesh, domains, neighbors)
        meristem_domain = np.unique(domains)[np.argmax(neighbor_counts)]
        coord = np.array(extract_domain(mesh, domains, meristem_domain).center_of_mass())
    else:
        coord = np.array(mesh.center_of_mass())

    meristem = int(domains[mesh.FindPoint(coord.tolist())])

    if return_coordinates:
        return meristem, coord
    return meristem


def extract_domain_data(
    point_data: pd.DataFrame,
    mesh: pv.PolyData,
    apex: NDArray[np.floating[Any]],
    meristem: int,
) -> pd.DataFrame:
    """Extract geometric and spatial data for each domain.

    Computes distance to boundary, distance to center of mass, angle from apex,
    surface area, maximum diameter, and meristem flag for each domain.

    Args:
        point_data: DataFrame with domain labels in 'domain' column
        mesh: PyVista mesh containing the domains
        apex: Apex coordinates as 3-element array
        meristem: Meristem domain index

    Returns:
        DataFrame with domain measurements and properties
    """
    domains_arr = np.unique(point_data.domain)
    domains_arr = domains_arr[~np.isnan(domains_arr)].astype(int)
    domain_data = pd.DataFrame(
        columns=[
            "domain",
            "distance_to_boundary",
            "distance_to_center_of_mass",
            "angle",
            "area",
            "max_diameter",
            "max_diameter_xy",
            "center_of_mass",
            "is_meristem",
        ],
        dtype=object,
    )

    for domain_id in domains_arr:
        # Get distance for closest boundary point to apex
        domain_mesh = get_domain(mesh, point_data, int(domain_id))
        domain_boundary = get_boundary_indices(domain_mesh)
        domain_boundary_coords = domain_mesh.points[domain_boundary]
        domain_boundary_dists = np.sqrt(np.sum((domain_boundary_coords - apex) ** 2, axis=1))
        distance_to_boundary: float = float(np.min(domain_boundary_dists))

        # Get distance for center of mass from apex
        center = np.array(domain_mesh.center_of_mass())
        distance_to_center_of_mass = float(np.sqrt(np.sum((center - apex) ** 2)))

        # Get domain size in terms of bounding boxes
        domain_coords = domain_mesh.points

        dists = cdist(domain_coords, domain_coords)
        max_diameter: float = float(np.max(dists))
        dists_xy = cdist(domain_coords[:, 1:], domain_coords[:, 1:])
        max_diameter_xy: float = float(np.max(dists_xy))

        # Get domain angle in relation to apex
        rel_pos = center - apex
        angle_val = float(np.arctan2(rel_pos[1], rel_pos[2]))  # angle in yz-plane
        if angle_val < 0:
            angle_val += 2.0 * np.pi
        angle_val *= 360 / (2.0 * np.pi)

        area = domain_mesh.area

        is_meristem = domain_id == meristem
        if is_meristem:
            angle_val = np.nan

        # Set data
        domain_data.loc[domain_id] = [  # type: ignore[invalid-assignment]  # pandas stubs
            int(domain_id),
            distance_to_boundary,
            distance_to_center_of_mass,
            angle_val,
            area,
            max_diameter,
            max_diameter_xy,
            tuple(center),
            is_meristem,
        ]
    domain_data = domain_data.infer_objects()
    domain_data = domain_data.sort_values(["is_meristem", "area"], ascending=False)
    return domain_data


def relabel_domains(
    point_data: pd.DataFrame,
    domain_data: pd.DataFrame,
    order: Literal["area", "max_diameter", "max_diameter_xy"] = "area",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel domains based on sorting criteria.

    Reorders domain labels based on size metrics, keeping meristem as label 0.

    Args:
        point_data: DataFrame with domain labels to update
        domain_data: DataFrame with domain measurements
        order: Sorting metric ('area', 'max_diameter', 'max_diameter_xy')

    Returns:
        Tuple of (updated point_data, updated domain_data) with relabeled domains
    """
    new_point_data = point_data.copy()
    new_domain_data = domain_data.copy()

    # Sort domains by the specified metric, keeping meristem first
    sort_columns = ["is_meristem", order]
    new_domain_data = new_domain_data.sort_values(sort_columns, ascending=False)

    domain_map: dict[int, int] = {}
    for ii in range(len(new_domain_data)):
        old_domain = new_domain_data.iloc[ii].domain
        domain_map[old_domain] = ii
        new_domain_data["domain"].iloc[ii] = ii

    for ii in domain_map:
        new_point_data.loc[point_data.domain == ii, "domain"] = domain_map[ii]

    return new_point_data, new_domain_data


# Backwards compatibility alias
extract_domaindata = extract_domain_data
remove_size = remove_small_domains
