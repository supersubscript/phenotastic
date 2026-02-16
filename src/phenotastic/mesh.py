from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import mahotas as mh
import networkx as nx
import numpy as np
import pymeshfix as pmf
import pyvista as pv
import scipy.optimize as opt
import tifffile as tiff
import vtk
from clahe import clahe
from imgmisc import autocrop, cut, get_resolution, to_uint8
from loguru import logger
from numpy.typing import NDArray
from pyacvd import clustering
from pymeshfix._meshfix import PyTMesh
from pystackreg import StackReg
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, zoom
from scipy.signal import wiener
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from skimage.segmentation import morphological_chan_vese

from phenotastic.misc import car2sph, coord_array, rotate

if TYPE_CHECKING:
    from phenotastic.phenomesh import PhenoMesh


def rotation_matrix_4x4(angles: Sequence[float], invert: bool = False) -> NDArray[np.floating[Any]]:
    """Generate 4x4 homogeneous rotation matrix.

    Rotations are applied in order beta-gamma-alpha (for historic reasons).

    Args:
        angles: Rotation angles in radians as 3-element array [alpha, beta, gamma]
        invert: If True, return inverse rotation matrix

    Returns:
        4x4 rotation matrix
    """
    alpha, beta, gamma = angles
    matrix_rot_x = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1],
        ],
    )
    matrix_rot_y = np.array(
        [
            [np.cos(beta), 0, np.sin(beta), 0],
            [0, 1, 0, 0],
            [-np.sin(beta), 0, np.cos(beta), 0],
            [0, 0, 0, 1],
        ],
    )
    matrix_rot_z = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0, 0],
            [np.sin(gamma), np.cos(gamma), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )

    if invert:
        matrix_rot = np.linalg.inv(np.matmul(np.matmul(matrix_rot_z, matrix_rot_y), matrix_rot_x))
    else:
        matrix_rot = np.matmul(np.matmul(matrix_rot_z, matrix_rot_y), matrix_rot_x)

    return matrix_rot


def paraboloid(
    parameters: tuple[float, float, float, float, float, float, float, float],
    sampling: tuple[int, int, int] | int = (200, 200, 200),
    bounds: tuple[float, float, float, float, float, float] | float | None = None,
) -> "PhenoMesh":
    """Generate meshed paraboloid surface.

    Creates a paraboloid mesh defined by: z = p0*x² + p1*y² + p2*x + p3*y + p4,
    rotated by angles [alpha, beta, gamma] (p[5:8]).

    Args:
        parameters: 8-element array [p0, p1, p2, p3, p4, alpha, beta, gamma] defining
            paraboloid coefficients and rotation angles
        sampling: Sampling resolution in XYZ dimensions
        bounds: Bounding box [X0,X1,Y0,Y1,Z0,Z1] or single float for symmetric bounds

    Returns:
        PhenoMesh of the paraboloid
    """
    p1, p2, p3, p4, p5, alpha, beta, gamma = parameters
    bounds_list: list[float]
    if bounds is None:
        bounds_list = [-2000, 2000] * 3
    elif isinstance(bounds, (float, int)):
        bounds_list = [-bounds, bounds] * 3
    else:
        bounds_list = list(bounds)

    sampling_list = [int(sampling)] * 3 if isinstance(sampling, (int, float)) else list(sampling)

    bounds_arr = np.array(bounds_list)

    # Get the bounding box corners in the transformed space
    corners = np.array(
        [
            [bounds_arr[0], bounds_arr[2], bounds_arr[4]],
            [bounds_arr[0], bounds_arr[2], bounds_arr[5]],
            [bounds_arr[0], bounds_arr[3], bounds_arr[4]],
            [bounds_arr[0], bounds_arr[3], bounds_arr[5]],
            [bounds_arr[1], bounds_arr[2], bounds_arr[4]],
            [bounds_arr[1], bounds_arr[2], bounds_arr[5]],
            [bounds_arr[1], bounds_arr[3], bounds_arr[4]],
            [bounds_arr[1], bounds_arr[3], bounds_arr[5]],
        ],
    )
    corners = rotate(corners, [alpha, beta, gamma], invert=False)

    bounds_final: list[float] = [
        float(min(corners[:, 0])),
        float(max(corners[:, 0])),
        float(min(corners[:, 1])),
        float(max(corners[:, 1])),
        float(min(corners[:, 2])),
        float(max(corners[:, 2])),
    ]

    # Generate the paraboloid mesh along the z-axis
    # vtkQuadric evaluates the quadric function:
    # F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2 + a3*x*y + a4*y*z + a5*x*z + a6*x + a7*y + a8*z + a9.
    quadric = vtk.vtkQuadric()
    quadric.SetCoefficients(p1, p2, 0, 0, 0, 0, p3, p4, -1, p5)
    sample = vtk.vtkSampleFunction()
    sample.SetSampleDimensions(sampling_list)
    sample.SetImplicitFunction(quadric)
    sample.SetModelBounds(bounds_final)
    sample.Update()

    # Rotate the mesh so that it is in the optimised orientation
    rotmat = rotation_matrix_4x4([alpha, beta, gamma], invert=True)
    trans = vtk.vtkMatrix4x4()
    for ii in range(rotmat.shape[0]):
        for jj in range(rotmat.shape[1]):
            trans.SetElement(ii, jj, rotmat[ii][jj])
    transmat = vtk.vtkMatrixToHomogeneousTransform()
    transmat.SetInput(trans)

    contour = vtk.vtkContourFilter()
    contour.SetInputData(sample.GetOutput())
    contour.Update()

    transfilter = vtk.vtkTransformPolyDataFilter()
    transfilter.SetInputData(contour.GetOutput())
    transfilter.SetTransform(transmat)
    transfilter.Update()

    output = pv.PolyData(transfilter.GetOutput())

    from phenotastic.phenomesh import PhenoMesh

    return PhenoMesh(output)


def create_mesh(contour: NDArray[Any], resolution: list[float] | None = None, step_size: int = 1) -> "PhenoMesh":
    """Generate mesh from binary 3D contour using marching cubes.

    Args:
        contour: Binary 3D array (bool, int, or float) defining the contour
        resolution: Spatial resolution in XYZ dimensions
        step_size: Step size for marching cubes algorithm

    Returns:
        PhenoMesh mesh
    """
    from phenotastic.phenomesh import PhenoMesh

    if resolution is None:
        resolution = [1, 1, 1]
    v, f, _, _ = marching_cubes(
        contour,
        0,
        spacing=list(resolution),
        step_size=step_size,
        allow_degenerate=False,
    )
    mesh = pv.PolyData(v, np.c_[[3] * len(f), f].ravel())
    return PhenoMesh(mesh)


def filter_curvature(
    mesh: pv.PolyData,
    curvature_threshold: tuple[float, float] | float,
    curvatures: NDArray[np.floating[Any]] | None = None,
) -> pv.PolyData:
    """Remove mesh vertices outside curvature threshold range.

    Args:
        mesh: PyVista PolyData mesh
        curvature_threshold: Tuple (min, max) defining valid curvature range,
            or single value for symmetric range
        curvatures: Optional curvature array. If None, computes mean curvature

    Returns:
        Filtered PyVista PolyData mesh
    """
    if isinstance(curvature_threshold, (int, float)):
        curvature_threshold = (-curvature_threshold, curvature_threshold)
    curvature = mesh.curvature() if curvatures is None else curvatures

    to_remove = (curvature < curvature_threshold[0]) | (curvature > curvature_threshold[1])
    mesh = mesh.remove_points(to_remove)[0]

    return mesh


def label_cellular_mesh(
    mesh: pv.PolyData,
    values: NDArray[Any],
    value_tag: str = "values",
    id_tag: str = "cell_id",
) -> pv.PolyData:
    """Assign values to mesh vertices based on cell IDs.

    Args:
        mesh: PyVista PolyData mesh with cell_id array
        values: Array of values to assign, indexed by cell ID
        value_tag: Name for the output value array in the mesh
        id_tag: Name of the cell ID array in the input mesh

    Returns:
        PyVista PolyData mesh with values assigned
    """
    mesh[value_tag] = np.zeros(mesh.n_points)
    for ii in np.unique(mesh[id_tag]):
        mesh[value_tag][mesh[id_tag] == ii] = values[ii]
    return mesh


def create_cellular_mesh(
    seg_img: NDArray[np.integer[Any]],
    resolution: list[float] | None = None,
    verbose: bool = True,
) -> "PhenoMesh":
    """Generate mesh from segmented 3D image with one mesh per cell.

    Args:
        seg_img: 3D segmentation image with integer labels for each cell
        resolution: Spatial resolution in XYZ dimensions
        verbose: Print progress information

    Returns:
        PhenoMesh combining all cell meshes
    """
    if resolution is None:
        resolution = [1, 1, 1]
    cells: list[pv.PolyData] = []
    labels = np.unique(seg_img)[1:]
    n_cells = len(labels)
    for c_idx, cell_id in enumerate(labels):
        if verbose:
            logger.info(f"Now meshing cell {c_idx} (label: {cell_id}) out of {n_cells}")
        cell_img, cell_cuts = autocrop(
            seg_img == cell_id,
            threshold=0,
            n=1,
            return_cuts=True,
            offset=[[2, 2], [2, 2], [2, 2]],
        )
        cell_volume = np.sum(cell_img > 0) * np.prod(resolution)

        v, f, _, _ = marching_cubes(
            cell_img,
            0,
            allow_degenerate=False,
            step_size=1,
            spacing=resolution,
        )
        v[:, 0] += cell_cuts[0, 0] * resolution[0]
        v[:, 1] += cell_cuts[1, 0] * resolution[1]
        v[:, 2] += cell_cuts[2, 0] * resolution[2]

        cell_mesh = pv.PolyData(v, np.ravel(np.c_[[[3]] * len(f), f]))
        cell_mesh["cell_id"] = np.full(
            fill_value=cell_id,
            shape=cell_mesh.n_points,
        )
        cell_mesh["volume"] = np.full(
            fill_value=cell_volume,
            shape=cell_mesh.n_points,
        )

        cells.append(cell_mesh)

    from phenotastic.phenomesh import PhenoMesh

    multi = pv.MultiBlock(cells)
    poly = pv.PolyData()
    for ii in range(multi.n_blocks):
        poly += multi.get(ii)

    return PhenoMesh(poly)


def _validate_data(data: NDArray[Any] | str) -> NDArray[Any]:
    if isinstance(data, str):
        data = tiff.imread(data)
        data = np.squeeze(data)
    return data


def _apply_stack_registration(data: NDArray[Any], fin: str | NDArray[Any], verbose: bool = True) -> NDArray[Any]:
    """Apply stack registration for slice alignment.

    Args:
        data: 3D or 4D image array
        fin: Input filename for logging
        verbose: Print processing progress

    Returns:
        Registered image array
    """
    if verbose:
        logger.info(f"Running stackreg for {fin}")
    pretype = data.dtype
    data = data.astype(float)
    sr = StackReg(StackReg.RIGID_BODY)
    if data.ndim > 3:
        trsf_mat = sr.register_stack(np.max(data, 1))
        for ii in range(data.shape[1]):
            data[:, ii] = sr.transform_stack(data[:, ii], tmats=trsf_mat)
    else:
        trsf_mat = sr.register_stack(data)
        data = sr.transform_stack(data, tmats=trsf_mat)
    data[data < 0] = 0
    data = data.astype(pretype)
    return data


def _apply_clahe_enhancement(
    data: NDArray[np.uint8],
    fin: str | NDArray[Any],
    clahe_window: Sequence[int] | None,
    clahe_clip_limit: float | None,
    verbose: bool,
) -> NDArray[np.uint8]:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement.

    Args:
        data: Input image data as uint8 array
        fin: Input filename or array (for verbose output)
        clahe_window: Window shape for CLAHE, if None will be computed
        clahe_clip_limit: Clip limit for CLAHE, if None will use Otsu threshold
        verbose: Flag to print processing steps

    Returns:
        CLAHE-enhanced image data
    """
    if verbose:
        logger.info(f"Running CLAHE for {fin}")
    if clahe_window is None:
        clahe_window = (np.array(data.shape) + 4) // 8
    if clahe_clip_limit is None:
        clahe_clip_limit = mh.otsu(data)
    return clahe(data, win_shape=clahe_window, clip_limit=clahe_clip_limit)


def _apply_wiener_filtering(data: NDArray[Any], fin: str | NDArray[Any], verbose: bool = True) -> NDArray[Any]:
    if verbose:
        logger.info(f"Running wiener filtering for {fin}")
    data = data.astype("float")
    if data.ndim > 3:
        for ii in range(data.shape[1]):
            data[:, ii] = wiener(data[:, ii])
        data = np.max(data, 1)
    else:
        data = wiener(data)
    return data


def contour(
    image: str | NDArray[Any],
    iterations: int = 25,
    smoothing: int = 1,
    masking_factor: float | NDArray[np.bool_] = 0.75,
    crop: bool = True,
    resolution: Sequence[float] | None = None,
    clahe_window: Sequence[int] | None = None,
    clahe_clip_limit: int | None = None,
    gaussian_sigma: list[float] | None = None,
    gaussian_iterations: int = 5,
    fill_slices: bool = True,
    chan_vese_lambda1: float = 1,
    chan_vese_lambda2: float = 1,
    register_stack: bool = True,
    target_resolution: list[float] | None = None,
    fill_inland_threshold: float | None = None,
    return_resolution: bool = False,
    verbose: bool = True,
) -> NDArray[np.bool_] | tuple[NDArray[np.bool_], Sequence[float]]:
    """Generate binary contour from 3D image using morphological active contours.

    Performs image preprocessing (CLAHE, Gaussian filtering), stack registration,
    and morphological Chan-Vese segmentation to extract object contour.

    Args:
        image: Input 3D image array or file path
        iterations: Number of morphological Chan-Vese iterations
        smoothing: Smoothing iterations per cycle
        masking: Initial mask threshold (fraction of Otsu threshold)
        crop: Automatically crop image to content
        resolution: Spatial resolution in XYZ (micrometers). Auto-detected if None
        clahe_window: CLAHE window size. Auto-calculated if None
        clahe_clip_limit: CLAHE clip limit. Auto-calculated if None
        gaussian_sigma: Gaussian filter sigma. Auto-calculated if None
        gaussian_iterations: Number of Gaussian smoothing iterations
        fill_slices: Fill holes in XY slices
        chan_vese_lambda1: Morphological Chan-Vese lambda1 parameter
        chan_vese_lambda2: Morphological Chan-Vese lambda2 parameter
        register_stack: Apply stack registration for slice alignment
        target_resolution: Target resolution for resampling
        fill_inland_threshold: Distance threshold for filling inland regions
        return_resolution: Return resolution along with contour
        verbose: Print processing progress

    Returns:
        Binary contour array, and optionally resolution tuple
    """

    if target_resolution is None:
        target_resolution = [0.5, 0.5, 0.5]
    if verbose:
        logger.info(f"Reading in data for {image}")

    data = _validate_data(image)

    if resolution is None:
        resolution = get_resolution(image) if isinstance(image, str) else [1, 1, 1]

    if any(np.less(resolution, 1e-3)):
        resolution = np.multiply(resolution, 1e6)
    if verbose:
        logger.info(f"Resolution for {image} is {resolution}")

    data = zoom(data, resolution / np.asarray(target_resolution), order=3)

    # Perform stack regularisation
    if register_stack:
        data = _apply_stack_registration(data, image, verbose)

    # Autocrop the image
    if crop:
        if verbose:
            logger.info(f"Running autocrop for {image}")
        offset = np.full((3, 2), 5)
        offset[0] = (10, 10)

        if data.ndim > 3:
            _, cuts = autocrop(
                np.max(data, 1),
                2 * mh.otsu(np.max(data, 1)),
                n=10,
                offset=offset,
                return_cuts=True,
            )
            data = cut(data, cuts)
        else:
            data = autocrop(data, 2 * mh.otsu(data), n=10, offset=offset)

    # Perform Wiener filtering
    data = _apply_wiener_filtering(data, image, verbose)
    data = to_uint8(data, False)

    # Performing CLAHE
    data = _apply_clahe_enhancement(data, image, clahe_window, clahe_clip_limit, verbose)

    # Performing Gaussian smoothing
    if gaussian_sigma is None:
        gaussian_sigma = [1, 1, 1]
    for _ii in range(gaussian_iterations):
        if verbose:
            logger.info(f"Smoothing out {image} with gaussian smoothing")
            data = gaussian_filter(data, sigma=gaussian_sigma)

    # Perform morphological chan-vese
    if verbose:
        logger.info(f"Running morphological chan-vese for {image}")
    mask: NDArray[np.bool_]
    if isinstance(masking_factor, (float, int)):
        mask = to_uint8(data, False) > masking_factor * mh.otsu(to_uint8(data, False))
    else:
        mask = masking_factor
    contour_result = morphological_chan_vese(
        data,
        iterations=iterations,
        init_level_set=mask,
        smoothing=smoothing,
        lambda1=chan_vese_lambda1,
        lambda2=chan_vese_lambda2,
    )

    # Postprocess the contour by filling etc.
    contour_result = fill_contour(contour_result, fill_xy=fill_slices, fill_zx_zy=False)

    if fill_inland_threshold is not None:
        if verbose:
            logger.info(f"Filling inland for {image}")
        contour_result = fill_inland(contour_result, int(fill_inland_threshold))

    if return_resolution:
        return contour_result, resolution
    return contour_result


def fill_beneath(contour: NDArray[np.bool_], mode: str = "bottom") -> NDArray[np.bool_]:
    """Fill contour beneath XY periphery projection.

    Args:
        contour: Binary 3D contour array
        mode: Fill mode ('bottom' or 'first')

    Returns:
        Filled contour array
    """
    cimg = contour.copy()
    cimg = np.pad(cimg, 1)

    # Get the indices for the first and last non-zero elements
    first = np.argmax(cimg, 0) if mode == "first" else np.zeros_like(cimg[0], dtype=np.uint16)
    last = cimg.shape[0] - np.argmax(cimg[::-1], 0) - 1
    last[last == cimg.shape[0] - 1] = 0

    # Fill the contour
    for ii in range(cimg.shape[1]):
        for jj in range(cimg.shape[2]):
            cimg[first[ii, jj] : last[ii, jj], ii, jj] = True
    cimg[:, last == 0] = False
    cimg = cimg[1:-1, 1:-1, 1:-1]

    return cimg


def fill_contour(
    contour: NDArray[np.bool_],
    fill_xy: bool = False,
    fill_zx_zy: bool = False,
    inplace: bool = False,
    zrange: Sequence[int] | None = None,
    xrange: Sequence[int] | None = None,
    yrange: Sequence[int] | None = None,
) -> NDArray[np.bool_] | None:
    """Fill holes in contour by closing edges and applying binary fill operations.

    Closes all edges except top, then fills holes. Note: may cause artifacts
    with significant curvature due to down-projection.

    Args:
        contour: Binary 3D contour array (Z, Y, X order, bottom to top)
        fill_xy: Fill holes in XY planes
        fill_zx_zy: Fill holes in ZX and ZY planes
        inplace: Modify contour in place
        zrange: Z-axis range to process [start, end]
        xrange: X-axis range to process [start, end]
        yrange: Y-axis range to process [start, end]

    Returns:
        Filled contour array (None if inplace=True)
    """
    new_contour = contour.copy() if not inplace else contour

    new_contour = np.pad(new_contour, 1, "constant", constant_values=1)
    xrange_list: list[int]
    yrange_list: list[int]
    zrange_list: list[int]

    if xrange is None:
        xrange_list = [0, new_contour.shape[2]]
    else:
        xrange_list = list(xrange)
        xrange_list[0] = xrange_list[0] - 1
    if yrange is None:
        yrange_list = [0, new_contour.shape[1]]
    else:
        yrange_list = list(yrange)
        yrange_list[0] = yrange_list[0] - 1
    if zrange is None:
        zrange_list = [0, new_contour.shape[0]]
    else:
        zrange_list = list(zrange)
        zrange_list[0] = zrange_list[0] - 1

    # Close all sides but top
    new_contour[-1] = 0  # top

    # Fill holes form in xz & yz planes.
    if fill_zx_zy:
        for ii in range(*yrange_list):
            new_contour[
                zrange_list[0] : zrange_list[1],
                ii,
                xrange_list[0] : xrange_list[1],
            ] = binary_fill_holes(
                new_contour[zrange_list[0] : zrange_list[1], ii, xrange_list[0] : xrange_list[1]],
            )
        for ii in range(*xrange_list):
            new_contour[
                zrange_list[0] : zrange_list[1],
                yrange_list[0] : yrange_list[1],
                ii,
            ] = binary_fill_holes(
                new_contour[zrange_list[0] : zrange_list[1], yrange_list[0] : yrange_list[1], ii],
            )

    # Remove edges again, also for top
    new_contour[0] = 0
    new_contour[-1] = 0
    new_contour[:, 0] = 0
    new_contour[:, -1] = 0
    new_contour[:, :, 0] = 0
    new_contour[:, :, -1] = 0

    if fill_xy:
        for ii in range(*zrange_list):
            new_contour[
                ii,
                yrange_list[0] : yrange_list[1],
                xrange_list[0] : xrange_list[1],
            ] = binary_fill_holes(
                new_contour[ii, yrange_list[0] : yrange_list[1], xrange_list[0] : xrange_list[1]],
            )

    new_contour = binary_fill_holes(new_contour)
    new_contour = new_contour[1:-1, 1:-1, 1:-1]

    return None if inplace else new_contour


def label_mesh(
    mesh: pv.PolyData,
    segm_img: NDArray[np.integer[Any]],
    resolution: list[float] | None = None,
    background: int = 0,
    mode: str = "point",
    inplace: bool = False,
) -> NDArray[np.integer[Any]] | None:
    """Label mesh vertices or faces using nearest voxel in segmented image.

    Args:
        mesh: PyVista PolyData mesh to label
        segm_img: 3D segmented image with integer labels
        resolution: Spatial resolution of segmented image
        background: Background label value to ignore
        mode: Labeling mode ('point' for vertices, 'face' for cell centers)
        inplace: Modify mesh in place

    Returns:
        Label array or None if inplace=True
    """
    if resolution is None:
        resolution = [1, 1, 1]
    vertex_flags = [
        "point",
        "points",
        "pts",
        "pt",
        "p",
        "vertex",
        "vertices",
        "vert",
        "verts",
        "v",
    ]
    triangle_flags = [
        "cell",
        "cells",
        "c",
        "triangle",
        "triangles",
        "tri",
        "tris",
        "polygon",
        "polygons",
        "poly",
        "polys",
    ]

    # Get the coordinates of the points in the image
    coords = coord_array(segm_img, tuple(resolution)).T
    img_raveled = segm_img.ravel()
    coords = coords[img_raveled != background]
    img_raveled = img_raveled[img_raveled != background]

    # Compute the closest coodinate to each point
    tree = cKDTree(coords)
    if mode.lower() in vertex_flags:
        closest = tree.query(mesh.points, k=1)[1]
    elif mode.lower() in triangle_flags:
        centers = mesh.cell_centers().points
        closest = tree.query(centers, k=1)[1]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Get the values of the closest points and either assign them to the mesh, or return them
    values = img_raveled[closest]
    if inplace:
        mesh["labels"] = values
        return None
    return values


def project2surface(
    mesh: pv.PolyData,
    int_img: NDArray[Any],
    distance_threshold: float,
    mask: NDArray[np.bool_] | None = None,
    resolution: list[float] | None = None,
    fct: Callable[..., Any] = np.sum,
    background: float = 0,
) -> NDArray[np.floating[Any]]:
    """Project image intensity values onto mesh surface.

    Aggregates intensity values within distance threshold of each vertex.

    Args:
        mesh: PyVista PolyData mesh
        int_img: 3D intensity image array
        distance_threshold: Maximum distance from surface for projection
        mask: Optional mask array for image
        resolution: Spatial resolution of intensity image
        fct: Aggregation function (currently only np.sum supported)
        background: Background value to ignore in image

    Returns:
        Array of projected values, one per mesh vertex
    """

    # Get the coordinates and values of the points in the image
    if resolution is None:
        resolution = [1, 1, 1]
    coords = coord_array(int_img, tuple(resolution)).T
    if mask is not None:
        int_img[~mask] = background
    img_raveled = int_img.ravel()

    # Don't consider background values
    coords = coords[img_raveled != background]
    img_raveled = img_raveled[img_raveled != background]

    # Filter out points clearly outside of the mesh to reduce computation time
    bounds: NDArray[np.floating[Any]] = np.reshape(mesh.bounds, (-1, 2))
    for ii, bound_pair in enumerate(bounds):
        within_bounds = (coords[:, ii] >= bound_pair[0]) & (coords[:, ii] <= bound_pair[1])
        img_raveled = img_raveled[within_bounds]
        coords = coords[within_bounds]

    # Get distance
    # Updating the normals is needed for the distance orientation
    mesh = mesh.compute_normals()
    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(mesh)
    dists = np.zeros((len(coords),))
    pts = np.zeros((len(coords), 3))
    for ii in range(len(coords)):
        dists[ii] = ipd.EvaluateFunctionAndGetClosestPoint(coords[ii], pts[ii])

    # Now filter out the values that are too far away
    if distance_threshold < 0:
        l1_filter = (dists > distance_threshold) & (dists < 0)
    else:
        l1_filter = (dists < distance_threshold) & (dists > 0)
    l1_coords = coords[l1_filter]
    l1_vals = img_raveled[l1_filter]

    # Get the closest point to each vertex and sum up the values for the vertex
    if fct is not np.sum:
        raise NotImplementedError("fct : Only np.sum is implemented for now.")

    tree = cKDTree(mesh.points)
    closest = tree.query(l1_coords, k=1)[1]
    values = np.zeros(mesh.n_points)

    for ii, val in enumerate(l1_vals):
        values[closest[ii]] += val

    return values


def remove_inland_under(
    mesh: pv.PolyData,
    contour: NDArray[np.bool_],
    threshold: int,
    resolution: list[float] | None = None,
    invert: bool = False,
) -> pv.PolyData:
    """Remove the part of the mesh that is under the contour.

    Args:
        mesh: Mesh to remove the inland part from.
        contour: Contour to use for the removal.
        threshold: Threshold distance from the contour XY periphery.
        resolution: Resolution of the image. Default is [1, 1, 1].
        invert: Invert the mesh normals. Default is False.

    Returns:
        Mesh with the inland part removed.
    """
    # Compute the domain of interest
    if resolution is None:
        resolution = [1, 1, 1]
    cont2d = np.max(contour, 0)
    cont2d = np.pad(cont2d, pad_width=1, constant_values=0, mode="constant")
    distance_map = distance_transform_edt(cont2d)
    distance_map = distance_map[1:-1, 1:-1]
    larger = np.array(np.where(distance_map > threshold)).T
    c = cont2d.astype(int)
    c[larger[:, 0], larger[:, 1]] = 2

    xycoords = np.divide(mesh.points[:, 1:].copy(), resolution[1:])
    xycoords = np.round(xycoords).astype(int)
    xycoords[xycoords < 0] = 0
    xycoords[xycoords[:, 0] > contour.shape[1] - 1, 0] = contour.shape[1] - 1
    xycoords[xycoords[:, 1] > contour.shape[2] - 1, 0] = contour.shape[2] - 1
    inside = c[xycoords[:, 0], xycoords[:, 1]] == 2
    indices = np.where(inside)[0]

    # Use ray-tracing to identify if the point is under the mesh surface or not
    under_indices: list[int] = []
    target = mesh.bounds[1] + 0.00001 if not invert else mesh.bounds[0] - 0.00001
    for ii in indices:
        pt = mesh.ray_trace(
            mesh.points[ii],
            [target, mesh.points[ii][1], mesh.points[ii][2]],
        )
        if pt[0].shape[0] > 1:
            under_indices.append(ii)
    under_indices_arr = np.array(under_indices)
    under = np.zeros(mesh.n_points, bool)
    if len(under_indices_arr) > 0:
        under[under_indices_arr] = True

    output = mesh.remove_points(under)[0]
    return output


def fill_inland(contour: NDArray[np.bool_], threshold_distance: int = 0) -> NDArray[np.bool_]:
    """Fill the contour based on the distance to the contour XY periphery.

    Args:
        contour: Contour to fill.
        threshold_distance: Threshold distance from the contour XY periphery.

    Returns:
        Filled contour.
    """
    # Get a maximum projection of the contour and compute the distane map
    cont2d = np.max(contour, 0)
    cont2d = np.pad(cont2d, pad_width=1, constant_values=0, mode="constant")
    distance_map = distance_transform_edt(cont2d)
    distance_map = distance_map[1:-1, 1:-1]
    larger = np.array(np.where(distance_map > threshold_distance)).T
    c = cont2d.astype(int)
    c[larger[:, 0], larger[:, 1]] = 2

    # Get the first and last indices in the Z-direction for each XY pixel
    first_occurence = np.argmax(contour, 0)
    last_occurence = contour.shape[0] - np.argmax(contour[::-1], 0) - 1
    last_occurence[last_occurence == contour.shape[0] - 1] = 0

    # Fill in the mask
    mask = np.zeros_like(contour)
    for ii in range(mask.shape[1]):
        for jj in range(mask.shape[2]):
            mask[first_occurence[ii, jj] : last_occurence[ii, jj], ii, jj] = True
    mask = mask & (c[1:-1, 1:-1] == 2)

    output = contour.copy()
    output[mask] = True
    filled = fill_contour(output, True)
    if filled is None:
        return output
    return filled


def repair_small(mesh: pv.PolyData, nbe: int | None = 100, refine: bool = True) -> pv.PolyData:
    """Repair small holes in a mesh based on the number of edges.

    Args:
        mesh: Mesh to repair.
        nbe: Number of edges to use for the repair. Default is 100.
        refine: Refine the mesh. Default is True.

    Returns:
        Repaired mesh.
    """
    mfix = PyTMesh(False)
    mfix.load_array(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    if nbe is None:
        nbe = -1
    mfix.fill_small_boundaries(nbe=nbe, refine=refine)
    vert, faces = mfix.return_arrays()

    output = pv.PolyData(vert, np.ravel(np.c_[[[3]] * len(faces), faces]))
    output = output.clean()
    output = output.triangulate()
    return output


def correct_bad_mesh(mesh: pv.PolyData, verbose: bool = True) -> pv.PolyData:
    """Correct a bad (non-manifold) mesh.

    Args:
        mesh: Input mesh.
        verbose: Flag to print out operation procedure.

    Returns:
        Corrected mesh.
    """
    try:
        from pymeshfix import _meshfix
    except ImportError:
        raise ImportError(
            "Package pymeshfix not found. Install to use this function.",
        ) from None

    new_poly = ecft(mesh, 0)
    nm = get_non_manifold_edges(new_poly)

    while nm.n_points > 0:
        if verbose:
            logger.info(f"Trying to remove {nm.GetNumberOfPoints()} points")

        # Create pymeshfix object from our mesh
        meshfix = _meshfix.PyTMesh()
        v, f = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        meshfix.load_array(v, f)

        # Remove smaller components
        meshfix.remove_smallest_components()
        v2, f2 = meshfix.return_arrays()
        f2 = np.hstack([np.append(len(ii), ii) for ii in f2])

        # Create new polydata from cleaned out mesh
        new_poly = pv.PolyData(v2, f2)
        new_poly = ecft(new_poly, 0)

        # If we still have non-manifold edges, force remove these points
        nm = get_non_manifold_edges(new_poly)
        nmpts = nm.points
        mpts = new_poly.points
        ptidx = np.array([np.where((mpts == ii).all(axis=1))[0][0] for ii in nmpts])

        mask = np.zeros((mpts.shape[0],), dtype=bool)
        if ptidx.shape[0] > 0:
            mask[ptidx] = True
        new_poly = new_poly.remove_points(mask)[0]

        new_poly = ecft(new_poly, 0)
        nm = get_non_manifold_edges(new_poly)

    new_poly = ecft(new_poly, 0)

    return new_poly


def remove_bridges(mesh: pv.PolyData, verbose: bool = True) -> pv.PolyData:
    """Remove triangles where all vertices are part of the mesh boundary.

    Args:
        mesh: Mesh to operate on.
        verbose: Flag to print processing steps.

    Returns:
        Mesh after bridge removal.
    """
    new_mesh = mesh

    while True:
        # Retrieve triangles on the border
        faces = new_mesh.faces.reshape(-1, 4)[:, 1:]
        f_flat = faces.ravel()
        boundary = get_boundary_points(new_mesh)
        border_faces = faces[
            np.unique(
                np.where(np.isin(f_flat, boundary))[0] // 3,
            )
        ]

        # Find pts to remove
        all_boundary = np.array([np.all(np.isin(ii, boundary)) for ii in border_faces])
        remove_pts = np.unique(border_faces[all_boundary].flatten())

        if verbose:
            logger.info(f"Removing {len(remove_pts)} points")
        if len(remove_pts) == 0:
            break

        # Actually remove
        mask = np.zeros((new_mesh.n_points,), dtype=bool)
        mask[remove_pts] = True

        new_mesh = new_mesh.remove_points(mask, keep_scalars=False)[0]
        new_mesh = ecft(new_mesh, 0)

    return new_mesh


def remove_normals(
    mesh: pv.PolyData,
    threshold_angle: float = 0,
    flip: bool = False,
    angle: str = "polar",
) -> pv.PolyData:
    """Remove points based on the point normal angle.

    Args:
        mesh: Mesh on which to operate.
        threshold_angle: Threshold for the polar angle (theta). Values smaller
            than this will be removed. Default = 0.
        flip: Flag to flip normal orientation. Default = False.
        angle: Type of angle to use ('polar' or 'azimuth').

    Returns:
        Mesh with the resulting vertices removed.
    """
    normals = mesh.point_normals.copy()
    if flip:
        normals *= -1.0
    normals = car2sph(normals) / (2.0 * np.pi) * 360.0

    if angle == "polar":
        angle_index = 1
    elif angle == "azimuth":
        angle_index = 2
    else:
        raise ValueError(
            "Parameter 'angle' can only take attributes 'polar' and 'azimuth'.",
        )

    to_remove = normals[:, angle_index] < threshold_angle
    new_mesh = mesh.remove_points(to_remove, keep_scalars=False)[0]
    return new_mesh


def smooth_boundary(
    mesh: pv.PolyData,
    iterations: int = 20,
    sigma: float = 0.1,
    inplace: bool = False,
) -> pv.PolyData | None:
    """Smooth the boundary of a mesh using Laplacian smoothing.

    Args:
        mesh: Mesh to smooth.
        iterations: Number of smoothing iterations. Default is 20.
        sigma: Smoothing sigma. Default is 0.1.
        inplace: Update mesh in-place. Default is False.

    Returns:
        Smoothed mesh, or None if inplace=True.
    """
    mesh = mesh.copy() if not inplace else mesh

    # Get boundary information and index correspondences
    boundary = get_boundary_edges(mesh)
    bdpts = boundary.points
    from_ = np.array([mesh.FindPoint(ii) for ii in bdpts])

    neighs: list[tuple[int, int]] = []
    for ii in range(boundary.n_points):
        pt_neighs = get_vertex_neighbors(boundary, ii, include_self=False)
        for jj in range(pt_neighs.shape[0]):
            neighs.append((ii, pt_neighs[jj]))

    # Find holes (cycles) in the mesh
    net = nx.Graph(neighs)
    cycles = nx.cycle_basis(net)
    cycles.sort(key=lambda x: len(x), reverse=True)
    cycles_arr = [np.array(ii) for ii in cycles]

    # Smooth boundary using Laplacian smoothing:
    # x_new = x_old - sigma * (x_old - mean(x_xneighs))
    new_pts_prev = bdpts.copy()
    new_pts_now = bdpts.copy()
    for _iter in range(iterations):
        new_pts_prev = new_pts_now.copy()
        for ii in range(len(cycles_arr)):
            for jj in range(len(cycles_arr[ii])):
                new_pts_now[cycles_arr[ii][jj]] = new_pts_prev[cycles_arr[ii][jj]] - sigma * (
                    new_pts_prev[cycles_arr[ii][jj]]
                    - np.mean(
                        np.array(
                            [
                                new_pts_prev[cycles_arr[ii][jj]],
                                new_pts_prev[cycles_arr[ii][jj - 1]],
                                new_pts_prev[cycles_arr[ii][(jj + 1) % len(cycles_arr[ii])]],
                            ],
                        ),
                        axis=0,
                    )
                )

    # Update coordinates
    for ii in range(len(cycles_arr)):
        mesh.points[from_[cycles_arr[ii]]] = new_pts_now[cycles_arr[ii]]

    return None if inplace else mesh


def process_mesh(
    mesh: pv.PolyData,
    hole_repair_threshold: int = 100,
    downscaling: float = 0.01,
    upscaling: float = 2,
    threshold_angle: float = 60,
    top_cut: str | tuple[float, float, float] = "center",
    tongues_radius: float | None = None,
    tongues_ratio: float = 4,
    smooth_iter: int = 200,
    smooth_relax: float = 0.01,
    curvature_threshold: float = 0.4,
    inland_threshold: float | None = None,
    contour: NDArray[np.bool_] | None = None,
) -> pv.PolyData:
    """Convenience function for postprocessing a mesh.

    Args:
        mesh: Mesh to process.
        hole_repair_threshold: Threshold for the hole repair algorithm.
        downscaling: Downscaling factor for the mesh.
        upscaling: Upscaling factor for the mesh.
        threshold_angle: Threshold for the polar angle (theta).
        top_cut: Top cut location.
        tongues_radius: Radius of the tongues.
        tongues_ratio: Ratio of the tongues.
        smooth_iter: Number of smoothing iterations.
        smooth_relax: Smoothing relaxation factor.
        curvature_threshold: Threshold for the curvature.
        inland_threshold: Threshold for the inland removal.
        contour: Contour to use for the inland removal.

    Returns:
        Processed mesh.
    """

    top_cut_tuple: tuple[float, float, float] = (
        (mesh.center[0], 0, 0) if top_cut == "center" else top_cut  # type: ignore[assignment]
    )

    # Scale the mest and repair small holes
    mesh = remesh(mesh, int(mesh.n_points * downscaling), sub=0)
    mesh = repair_small(mesh, hole_repair_threshold)

    # Remove vertices based on the vertex normal angle
    if threshold_angle:
        mesh.rotate_y(-90)
        mesh = remove_normals(
            mesh,
            threshold_angle=threshold_angle,
            angle="polar",
        )
        mesh.rotate_y(90)
        mesh = make_manifold(mesh, hole_repair_threshold)
        mesh = mesh.extract_largest()
        mesh.clear_data()
        mesh = correct_normal_orientation_topcut(mesh, top_cut_tuple)

    # Remove vertices based on whether they are "inside the contour"
    if inland_threshold is not None and contour is not None:
        mesh = remove_inland_under(mesh, contour, threshold=int(inland_threshold))
        mesh = mesh.extract_largest()
        mesh = repair_small(mesh, hole_repair_threshold)
    mesh = ecft(mesh, hole_repair_threshold)

    # Remove "tongues" from the mesh
    if tongues_radius is not None:
        mesh = remove_tongues(
            mesh,
            radius=tongues_radius,
            threshold=tongues_ratio,
            hole_edges=hole_repair_threshold,
        )

    # General post-processing. Smooth the mesh, remove small holes, and regularise the faces.
    mesh = mesh.extract_largest()
    mesh = repair_small(mesh, hole_repair_threshold)
    mesh = mesh.smooth(smooth_iter, smooth_relax)
    mesh = remesh(mesh, int(upscaling * mesh.n_points))
    result = smooth_boundary(mesh, smooth_iter, smooth_relax)
    if result is not None:
        mesh = result

    return mesh


def remove_tongues(
    mesh: pv.PolyData,
    radius: float,
    threshold: float = 6,
    hole_edges: int = 100,
    verbose: bool = True,
) -> pv.PolyData:
    """Remove "tongues" in mesh.

    All boundary points within a given radius are considered. The ones where the
    fraction of the distance along the boundary, as divided by the euclidean
    distance, is greater than the given threshold.

    Args:
        mesh: Mesh to operate on.
        radius: Radius for boundary point neighbourhood.
        threshold: Threshold for fraction between boundary distance and euclidean distance.

    Returns:
        Resulting mesh.
    """
    while True:
        # Get boundary information and index correspondences
        boundary = get_boundary_edges(mesh)
        bdpts = boundary.points
        from_ = np.array([mesh.FindPoint(ii) for ii in bdpts])

        all_neighs = get_vertex_neighbors_all(mesh, include_self=False)
        all_edges: list[tuple[int, int]] = []
        for ii, pt_neighs in enumerate(all_neighs):
            for _jj, neigh in enumerate(pt_neighs):
                all_edges.append((ii, neigh))
        all_edges_arr = np.array(all_edges)

        weighted_all_edges = np.c_[
            all_edges_arr,
            np.sum(
                (mesh.points[all_edges_arr[:, 0]] - mesh.points[all_edges_arr[:, 1]]) ** 2,
                1,
            )
            ** 0.5,
        ]
        all_net = nx.Graph()
        all_net.add_weighted_edges_from(weighted_all_edges)

        # Find the cycles, i.e. the different boundaries we have
        neighs_list: list[tuple[int, int]] = []
        for ii in range(boundary.n_points):
            pt_neighs = get_vertex_neighbors(boundary, ii, include_self=False)
            for jj in range(pt_neighs.shape[0]):
                neighs_list.append((ii, pt_neighs[jj]))
        neighs_arr = np.array(neighs_list)

        weighted_edges = np.c_[
            neighs_arr,
            np.sum(
                (bdpts[neighs_arr[:, 0]] - bdpts[neighs_arr[:, 1]]) ** 2,
                1,
            )
            ** 0.5,
        ]
        bdnet = nx.Graph()
        bdnet.add_weighted_edges_from(weighted_edges)

        cycles = nx.cycle_basis(bdnet)
        cycles.sort(key=lambda x: len(x), reverse=True)
        cycles_int = [np.array(ii, dtype=int) for ii in cycles]

        # Loop over the cycles and find boundary points within radius
        to_remove: list[int] = []
        for ii, cycle in enumerate(cycles_int):
            if verbose:
                logger.info(f"Running cycle {ii} with {len(cycle)} points")
            cpts = bdpts[cycle]

            # Get the boundary points (in same loop) within a certain radius
            tree = cKDTree(cpts)
            neighs_ball = tree.query_ball_point(cpts, radius)
            neighs_filtered = [np.array(neigh) for neigh in neighs_ball]
            neighs_filtered = [neigh[neigh != idx] for idx, neigh in enumerate(neighs_filtered)]

            # Get shortest geodesic path from every point in the cycle to all of it's
            # neighbours within the radius
            bd_dists: list[NDArray[np.floating[Any]]] = []
            int_dists: list[NDArray[np.floating[Any]]] = []
            for jj in range(len(cpts)):
                bd_path_lengths: list[float] = []
                int_path_lengths: list[float] = []
                for kk in range(len(neighs_filtered[jj])):
                    bd_length = nx.shortest_path_length(
                        bdnet,
                        source=cycle[jj],
                        target=cycle[neighs_filtered[jj][kk]],
                        weight="weight",
                    )
                    int_length = nx.shortest_path_length(
                        all_net,
                        source=from_[cycle[jj]],
                        target=from_[cycle[neighs_filtered[jj][kk]]],
                        weight="weight",
                    )

                    bd_path_lengths.append(bd_length)
                    int_path_lengths.append(int_length)

                bd_dists.append(np.array(bd_path_lengths))
                int_dists.append(np.array(int_path_lengths))

            frac = [bd_dists[jj] / int_dists[jj] for jj in range(len(neighs_filtered))]

            # Find which ones to (possibly) remove
            removal_anchors: list[tuple[int, Any]] = []
            for kk in range(len(frac)):
                for jj in range(len(frac[kk])):
                    if frac[kk][jj] > threshold:
                        removal_anchors.append((kk, neighs_filtered[kk][jj]))
            removal_anchors_arr = np.array(removal_anchors)

            # Recalculate the geodesic path between two points
            for jj in range(len(removal_anchors_arr)):
                gdpts = nx.shortest_path(
                    all_net,
                    source=from_[cycle[removal_anchors_arr[jj][0]]],
                    target=from_[cycle[removal_anchors_arr[jj][1]]],
                    weight="weight",
                )
                gdpts_arr = np.array(gdpts, dtype="int")
                to_remove.extend(gdpts_arr.tolist())

        to_remove_arr = np.unique(to_remove)

        if len(to_remove_arr) == 0:
            break

        # Remove points and do some cleanup
        mesh = mesh.remove_points(to_remove_arr, keep_scalars=False)[0]
        mesh = repair_small(mesh, hole_edges)
        mesh = make_manifold(mesh, hole_edges)
        mesh = ecft(mesh, hole_edges)

    mesh = mesh.clean()
    return mesh


def repair(mesh: pv.PolyData) -> pv.PolyData:
    """Repair a mesh using MeshFix."""
    tmp = pmf.MeshFix(mesh)
    tmp.repair(True)
    return tmp.mesh


def remesh(mesh: pv.PolyData, n: int, sub: int = 3) -> pv.PolyData:
    """Regularise the mesh faces using the ACVD algorithm.

    Args:
        mesh: The mesh to regularise.
        n: The number of clusters (i.e. output number of faces) to use.
        sub: The number of subdivisions to use when clustering.

    Returns:
        The regularised mesh.
    """
    clus = clustering.Clustering(mesh)
    clus.subdivide(sub)
    clus.cluster(n)
    output = clus.create_mesh()
    output = output.clean()
    return output


def make_manifold(mesh: pv.PolyData, hole_edges: int = 300) -> pv.PolyData:
    """Make a mesh manifold by removing non-manifold edges."""
    mesh = mesh.copy()
    edges = mesh.extract_feature_edges(
        boundary_edges=False,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=True,
    )
    while edges.n_points > 0:
        to_remove = [mesh.FindPoint(pt) for pt in edges.points]
        logger.info(f"Removing {len(to_remove)} points")
        mesh = mesh.remove_points(to_remove)[0]
        mesh = mesh.extract_largest()
        mesh = repair_small(mesh, nbe=hole_edges)
        mesh = mesh.clean()
        edges = mesh.extract_feature_edges(
            boundary_edges=False,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=True,
        )

    return mesh


def drop_skirt(mesh: pv.PolyData, maxdist: float, flip: bool = False) -> pv.PolyData:
    """Downprojects the boundary to the lowest point in the z-direction.

    Args:
        mesh: Mesh to operate on.
        maxdist: Distance in z-direction from the lowest point in the mesh to consider.
        flip: If True, flip the direction.

    Returns:
        Mesh with boundary downprojected.
    """
    lowest = mesh.bounds[int(flip)]
    boundary = get_boundary_edges(mesh)

    mpts = mesh.points
    bdpts = boundary.points
    idx_in_parent = np.array([mesh.FindPoint(ii) for ii in bdpts])

    to_adjust = idx_in_parent[bdpts[:, 0] - lowest < maxdist]
    mpts[to_adjust, 0] = lowest

    new_mesh = pv.PolyData(mpts, mesh.faces)

    return new_mesh


def get_boundary_points(mesh: pv.PolyData) -> NDArray[np.intp]:
    """Get vertex indices of points in the boundary."""
    boundary = get_boundary_edges(mesh)
    bdpts = boundary.points
    indices = np.array([mesh.FindPoint(ii) for ii in bdpts])

    return indices


def remesh_decimate(
    mesh: pv.PolyData,
    iters: int,
    upfactor: float = 2,
    downfactor: float = 0.5,
    verbose: bool = True,
) -> pv.PolyData:
    """Iterative remeshing/decimation.

    Args:
        mesh: Mesh to operate on.
        iters: Number of iterations.
        upfactor: Factor with which to upsample.
        downfactor: Factor with which to downsample.
        verbose: Flag for whether to print operation steps.

    Returns:
        Processed mesh.
    """
    for _ii in range(iters):
        mesh = correct_bad_mesh(mesh, verbose=verbose)
        mesh = ecft(mesh, 0)

        mesh = remesh(mesh, mesh.GetNumberOfPoints() * 2)
        mesh = mesh.compute_normals(inplace=False)
        mesh = mesh.decimate(
            0.5,
            volume_preservation=True,
            normals=True,
            inplace=False,
        )
        mesh = ecft(mesh, 0)

    return mesh


def get_non_manifold_edges(mesh: pv.PolyData) -> pv.PolyData:
    """Get non-manifold edges."""
    edges = mesh.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    return edges


def get_boundary_edges(mesh: pv.PolyData) -> pv.PolyData:
    """Get boundary edges."""
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    )
    return edges


def get_manifold_edges(mesh: pv.PolyData) -> pv.PolyData:
    """Get manifold edges."""
    edges = mesh.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=True,
    )
    return edges


def get_feature_edges(mesh: pv.PolyData, angle: float = 30) -> pv.PolyData:
    """Get feature edges defined by given angle."""
    edges = mesh.extract_feature_edges(
        feature_angle=angle,
        boundary_edges=False,
        non_manifold_edges=False,
        feature_edges=True,
        manifold_edges=False,
    )
    return edges


def get_vertex_neighbors(mesh: pv.PolyData, index: int, include_self: bool = True) -> NDArray[np.intp]:
    """Get the indices of the vertices connected to a given vertex.

    Args:
        mesh: Mesh to operate on.
        index: Index of the vertex.
        include_self: Whether to include the vertex itself in the list of neighbors.

    Returns:
        Array of indices of the connected vertices.
    """
    connected_vertices: list[int] = []
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

    return np.unique(connected_vertices)


def get_vertex_neighbors_all(mesh: pv.PolyData, include_self: bool = True) -> list[NDArray[np.intp]]:
    """Get all vertex neighbors.

    Args:
        mesh: Mesh to operate on.
        include_self: Whether to include the vertex itself in the list of neighbors.

    Returns:
        List of arrays of indices of the connected vertices.
    """
    connectivities: list[NDArray[np.intp]] = []
    for ii in range(mesh.n_points):
        connectivities.append(get_vertex_neighbors(mesh, ii, include_self))

    return connectivities


def correct_normal_orientation_topcut(mesh: pv.PolyData, origin: tuple[float, float, float]) -> pv.PolyData:
    """Correct normal orientation of a mesh by flipping normals if needed."""
    mesh.clear_data()
    if mesh.clip(normal="-x", origin=origin).point_normals[:, 0].sum() > 0:
        mesh.flip_normals()
    return mesh


def ecft(mesh: pv.PolyData, hole_edges: int = 300, inplace: bool = False) -> pv.PolyData:
    """Perform ExtractLargest, Clean, FillHoles, and TriFilter operations.

    Args:
        mesh: Mesh to operate on.
        hole_edges: Size of holes to fill.
        inplace: Flag for performing operation in-place.

    Returns:
        Mesh after operation.
    """
    new_mesh = mesh if inplace else mesh.copy()

    new_mesh = new_mesh.extract_largest()
    new_mesh = new_mesh.clean()
    new_mesh = repair_small(mesh, nbe=hole_edges)
    new_mesh = new_mesh.triangulate()
    new_mesh.clean()

    return new_mesh


# Alias for backwards compatibility
ECFT = ecft


def define_meristem(
    mesh: pv.PolyData,
    method: str = "central_mass",
    res: tuple[float, float, float] = (1, 1, 1),
    return_coord: bool = False,
) -> int | tuple[int, NDArray[np.floating[Any]]]:
    """Determine which domain corresponds to the meristem.

    Args:
        mesh: Mesh to operate on.
        method: Method for defining the meristem to use.
        res: Resolution of the dimensions.
        return_coord: If True, return coordinates as well.

    Returns:
        Domain index of the meristem, and optionally the center coordinates.
    """
    ccoord = np.zeros((3,))
    if method == "central_mass":
        com = vtk.vtkCenterOfMass()
        com.SetInputData(mesh)
        com.Update()
        ccoord = np.array(com.GetCenter())
    elif method == "central_bounds":
        ccoord = np.mean(np.reshape(mesh.GetBounds(), (3, 2)), axis=1)

    m_idx = np.argmin(((mesh.points - ccoord) ** 2).sum(1) ** 0.5)
    meristem = int(mesh["domains"][m_idx])
    if return_coord:
        return meristem, ccoord
    return meristem


def erode(mesh: pv.PolyData, iterations: int = 1) -> pv.PolyData:
    """Erode the mesh by removing boundary points iteratively."""
    mesh = mesh.copy()
    for _iter in range(iterations):
        if mesh.n_points == 0:
            break
        mesh = mesh.remove_points(get_boundary_points(mesh))[0]
    return mesh


def fit_paraboloid(
    data: NDArray[np.floating[Any]],
    init: list[float] | None = None,
    return_success: bool = False,
) -> NDArray[np.floating[Any]] | tuple[NDArray[np.floating[Any]], bool]:
    """Fit a paraboloid to arbitrarily oriented 3D data.

    The paraboloid data can by oriented along an arbitrary axis --
    not necessarily x, y, z. The function rotates the data points and returns
    the rotation angles along the x, y, z axis.

    Args:
        data: Data to fit the paraboloid to.
        init: Initial parameters for the paraboloid.
        return_success: If True, return success status as well.

    Returns:
        Parameters after optimisation, and optionally success status.
    """

    if init is None:
        init = [1, 1, 1, 1, 1, 0, 0, 0]

    def errfunc(p: Sequence[float], coord: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        p1, p2, p3, p4, p5, alpha, beta, gamma = p
        coord = rotate(coord, [alpha, beta, gamma])

        x, y, z = np.array(coord).T
        return np.abs(p1 * x**2.0 + p2 * y**2.0 + p3 * x + p4 * y + p5 - z)

    popt, _1, _2, _3, _4 = opt.leastsq(
        errfunc,
        init,
        args=(data,),
        full_output=True,
    )
    if return_success:
        return popt, _4 in [1, 2, 3, 4]
    return popt


def get_vertex_cycles(mesh: pv.PolyData) -> list[list[int]]:
    """Find cycles (holes/boundaries) in a mesh.

    Args:
        mesh: Mesh to operate on.

    Returns:
        List of cycles, each cycle is a list of vertex indices.
    """
    all_neighs = get_vertex_neighbors_all(mesh, include_self=True)
    pairs: list[tuple[int, int]] = []
    for ii in range(mesh.n_points):
        for pp in all_neighs[ii]:
            pairs.append((ii, int(pp)))
    net = nx.Graph(pairs)
    cycles: list[list[int]] = nx.cycle_basis(net)
    cycles.sort(key=lambda x: len(x), reverse=True)
    return cycles


def correct_normal_orientation(mesh: pv.PolyData, relative: str = "x", inplace: bool = False) -> pv.PolyData | None:
    """Correct the orientation of the normals of a mesh."""
    mesh = mesh if inplace else mesh.copy()
    normals = mesh.point_normals

    if (
        (relative == "x" and normals[:, 0].sum() > 0)
        or (relative == "y" and normals[:, 1].sum() > 0)
        or (relative == "z" and normals[:, 2].sum() > 0)
    ):
        mesh.flip_normals()

    return None if inplace else mesh


def fit_paraboloid_mesh(
    mesh: pv.PolyData,
    return_coord: bool = False,
) -> NDArray[np.floating[Any]] | tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Fit a paraboloid to a mesh.

    Args:
        mesh: Mesh to fit paraboloid to.
        return_coord: If True, return apex coordinates as well.

    Returns:
        Parameters for the paraboloid, and optionally the apex coordinates.
    """
    popt = fit_paraboloid(mesh.points)
    if isinstance(popt, tuple):
        popt = popt[0]
    if return_coord:
        apex = compute_paraboloid_apex(popt)
        return popt, apex
    return popt


def compute_paraboloid_apex(parameters: Sequence[float]) -> NDArray[np.floating[Any]]:
    """Return the apex coordinates of a paraboloid.

    Args:
        parameters: 8-tuple of parameters defining the paraboloid.

    Returns:
        Coordinates for the paraboloid apex.
    """
    p1, p2, p3, p4, p5, alpha, beta, gamma = parameters
    x = -p3 / (2.0 * p1)
    y = -p4 / (2.0 * p2)
    z = p1 * x**2.0 + p2 * y**2.0 + p3 * x + p4 * y + p5

    coords = rotate(
        np.array(
            [
                [x, y, z],
            ],
        ),
        [alpha, beta, gamma],
        True,
    )[0]

    return coords
