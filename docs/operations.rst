==========
Operations
==========

This page documents all available pipeline operations and their parameters.

Image/Contour Operations
------------------------

contour
~~~~~~~

Generate binary contour from 3D image using morphological active contours.

**Parameters:**

- ``iterations`` (int, default: 25): Number of morphological Chan-Vese iterations
- ``smoothing`` (int, default: 1): Smoothing iterations per cycle
- ``masking_factor`` (float, default: 0.75): Initial mask threshold factor
- ``target_resolution`` (list[float], default: [0.5, 0.5, 0.5]): Target resolution for resampling
- ``gaussian_sigma`` (list[float], default: [1.0, 1.0, 1.0]): Gaussian filter sigma
- ``gaussian_iterations`` (int, default: 5): Number of Gaussian smoothing iterations
- ``register_stack`` (bool, default: True): Apply stack registration
- ``chan_vese_lambda1`` (float, default: 1.0): Chan-Vese lambda1 parameter
- ``chan_vese_lambda2`` (float, default: 1.0): Chan-Vese lambda2 parameter
- ``fill_slices`` (bool, default: True): Fill holes in XY slices
- ``crop`` (bool, default: True): Automatically crop image

create_mesh
~~~~~~~~~~~

Create mesh from contour using marching cubes.

**Parameters:**

- ``step_size`` (int, default: 1): Step size for marching cubes

create_cellular_mesh
~~~~~~~~~~~~~~~~~~~~

Create mesh from segmented image (one mesh per cell).

**Parameters:**

- ``verbose`` (bool, default: True): Print progress information

Mesh Processing Operations
--------------------------

smooth
~~~~~~

Laplacian smoothing.

**Parameters:**

- ``iterations`` (int, default: 100): Number of smoothing iterations
- ``relaxation_factor`` (float, default: 0.01): Relaxation factor (0-1)
- ``feature_smoothing`` (bool, default: False): Smooth along features
- ``boundary_smoothing`` (bool, default: True): Smooth boundary edges

smooth_taubin
~~~~~~~~~~~~~

Taubin smoothing (less shrinkage than Laplacian).

**Parameters:**

- ``iterations`` (int, default: 100): Number of smoothing iterations
- ``pass_band`` (float, default: 0.1): Pass band for filter (0-2)
- ``feature_smoothing`` (bool, default: False): Smooth along features
- ``boundary_smoothing`` (bool, default: True): Smooth boundary edges

smooth_boundary
~~~~~~~~~~~~~~~

Smooth only boundary edges.

**Parameters:**

- ``iterations`` (int, default: 20): Number of smoothing iterations
- ``sigma`` (float, default: 0.1): Smoothing sigma

remesh
~~~~~~

Regularize faces using ACVD algorithm.

**Parameters:**

- ``n_clusters`` (int, default: 10000): Target number of faces
- ``subdivisions`` (int, default: 3): Number of subdivisions for clustering

decimate
~~~~~~~~

Reduce mesh complexity by removing faces.

**Parameters:**

- ``target_reduction`` (float, default: 0.5): Fraction of faces to remove (0-1)
- ``volume_preservation`` (bool, default: True): Preserve mesh volume

subdivide
~~~~~~~~~

Increase mesh resolution by subdividing faces.

**Parameters:**

- ``n_subdivisions`` (int, default: 1): Number of subdivision iterations
- ``subfilter`` (str, default: "linear"): Subdivision filter ('linear', 'butterfly', 'loop')

repair_holes
~~~~~~~~~~~~

Fill small holes in the mesh.

**Parameters:**

- ``max_hole_edges`` (int, default: 100): Maximum hole size to fill (in edges)
- ``refine`` (bool, default: True): Refine the filled region

repair
~~~~~~

Full mesh repair using MeshFix.

**Parameters:** None

make_manifold
~~~~~~~~~~~~~

Remove non-manifold edges.

**Parameters:**

- ``hole_edges`` (int, default: 300): Size of holes to fill after removal

filter_curvature
~~~~~~~~~~~~~~~~

Remove vertices outside curvature threshold range.

**Parameters:**

- ``threshold`` (float or list[float], default: 0.4): Single value for symmetric range [-t, t] or [min, max] list

remove_normals
~~~~~~~~~~~~~~

Remove vertices based on normal angle.

**Parameters:**

- ``threshold_angle`` (float, default: 60.0): Angle threshold in degrees
- ``flip`` (bool, default: False): Flip normal orientation before filtering
- ``angle_type`` (str, default: "polar"): Type of angle ('polar' or 'azimuth')

remove_bridges
~~~~~~~~~~~~~~

Remove triangles where all vertices are on the boundary.

**Parameters:** None

remove_tongues
~~~~~~~~~~~~~~

Remove tongue-like artifacts.

**Parameters:**

- ``radius`` (float, default: 50.0): Radius for boundary point neighborhood
- ``threshold`` (float, default: 6.0): Threshold for boundary/euclidean distance ratio
- ``hole_edges`` (int, default: 100): Size of holes to fill after removal

extract_largest
~~~~~~~~~~~~~~~

Keep only the largest connected component.

**Parameters:** None

clean
~~~~~

Remove degenerate cells.

**Parameters:**

- ``tolerance`` (float, default: None): Tolerance for point merging

triangulate
~~~~~~~~~~~

Convert all faces to triangles.

**Parameters:** None

compute_normals
~~~~~~~~~~~~~~~

Compute surface normals.

**Parameters:**

- ``flip`` (bool, default: False): Flip all normals
- ``consistent`` (bool, default: True): Make normals consistent
- ``auto_orient`` (bool, default: False): Orient normals outward

flip_normals
~~~~~~~~~~~~

Flip all surface normals.

**Parameters:** None

correct_normal_orientation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Correct normal orientation relative to an axis.

**Parameters:**

- ``relative`` (str, default: "x"): Axis to use for orientation ('x', 'y', 'z')

rotate
~~~~~~

Rotate mesh around an axis.

**Parameters:**

- ``axis`` (str, default: "x"): Axis to rotate around ('x', 'y', 'z')
- ``angle`` (float, default: 0.0): Rotation angle in degrees

clip
~~~~

Clip mesh with a plane.

**Parameters:**

- ``normal`` (str or list[float], default: "x"): Plane normal ('x', 'y', 'z', '-x', '-y', '-z') or [nx, ny, nz]
- ``origin`` (list[float], default: None): Point on clipping plane
- ``invert`` (bool, default: True): Invert clipping direction

erode
~~~~~

Erode mesh by removing boundary points.

**Parameters:**

- ``iterations`` (int, default: 1): Number of erosion iterations

ecft
~~~~

ExtractLargest, Clean, FillHoles, Triangulate (combined operation).

**Parameters:**

- ``hole_edges`` (int, default: 300): Size of holes to fill

Domain Operations
-----------------

compute_curvature
~~~~~~~~~~~~~~~~~

Compute mesh curvature.

**Parameters:**

- ``curvature_type`` (str, default: "mean"): Type of curvature ('mean', 'gaussian', 'minimum', 'maximum')

filter_scalars
~~~~~~~~~~~~~~

Apply filter to curvature field.

**Parameters:**

- ``function`` (str, default: "median"): Filter function ('median', 'mean', 'minmax', 'maxmin')
- ``iterations`` (int, default: 1): Number of filter iterations

segment_domains
~~~~~~~~~~~~~~~

Create domains via steepest ascent on curvature field.

**Parameters:**

- ``curvature_type`` (str, default: None): Curvature type to compute if not already in context

merge_angles
~~~~~~~~~~~~

Merge domains within angular threshold from meristem.

**Parameters:**

- ``threshold`` (float, default: 20.0): Angular threshold in degrees
- ``meristem_method`` (str, default: "center_of_mass"): Method for calculating meristem center

merge_distance
~~~~~~~~~~~~~~

Merge domains within spatial distance threshold.

**Parameters:**

- ``threshold`` (float, default: 50.0): Distance threshold
- ``metric`` (str, default: "euclidean"): Distance metric ('euclidean' or 'geodesic')
- ``method`` (str, default: "center_of_mass"): Method for calculating domain center

merge_small
~~~~~~~~~~~

Merge small domains to their largest neighbor.

**Parameters:**

- ``threshold`` (int, default: 100): Size threshold for merging
- ``metric`` (str, default: "points"): Size metric ('points' or 'area')
- ``mode`` (str, default: "border"): Merge strategy ('border' or 'area')

merge_engulfing
~~~~~~~~~~~~~~~

Merge domains mostly encircled by a neighbor.

**Parameters:**

- ``threshold`` (float, default: 0.9): Fraction of boundary that must be shared (0-1)

merge_disconnected
~~~~~~~~~~~~~~~~~~

Connect domains isolated from meristem.

**Parameters:**

- ``meristem_method`` (str, default: "center_of_mass"): Method for identifying meristem

merge_depth
~~~~~~~~~~~

Merge domains with similar depth values.

**Parameters:**

- ``threshold`` (float, default: 0.0): Maximum depth difference for merging
- ``mode`` (str, default: "max"): Aggregation mode ('min', 'max', 'median', 'mean')
- ``exclude_boundary`` (bool, default: False): Exclude boundary vertices from calculation
- ``min_points`` (int, default: 0): Minimum border points required for merging

define_meristem
~~~~~~~~~~~~~~~

Identify the meristem domain.

**Parameters:**

- ``method`` (str, default: "center_of_mass"): Method for meristem identification

extract_domaindata
~~~~~~~~~~~~~~~~~~

Extract geometric measurements for each domain.

**Parameters:** None

**Output:**

Creates a DataFrame with domain measurements including:

- Domain index
- Area
- Centroid coordinates
- Distance from apex
- Angular position
