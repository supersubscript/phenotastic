======================
Meristem Phenotyper 3D
======================

*Meristem Phenotyper 3D* is a python class and collection of python functions 
for the automated extraction of features of a shoot apical meristem (SAM) from a 
stack of confocal microscope images.
Its main features are:

* Extraction of the SAMs surface using active contour without edges [1]_ (ACWE)
  or a threshold based method.
* Separation of the SAMs features into primordiae and the meristem and fitting
  them with a spheres or a paraboloid.
* Evaluation of the features in terms of location, size and divergence angle.

Dependencies
============

This software is written for Python 2.7.6

Used packages are:

* SciPy 0.19.0
* NumPy 1.13.0
* VTK 5.8.0
* pandas 0.17.1
* tissueviewer
* `morphsnakes <https://github.com/pmneila/morphsnakes>`_

Installation
============
Download *Meristem Phenotyper 3D* for example by cloning the repository.
Add *automated_phenotyping.py* to your pythonpath or put it in the same folder
as your script and import it.

.. code:: python

  import Meristem_Phenotyper_3D as ap

Batch Processor
===============
The *batch processor* is a graphical user interface, that uses *Meristem
Phenotyper 3D*. It has all functions of the tool and can run several files in
sequence.

.. figure:: images/batch_processor_screenshot.png
   :height: 256
   :width: 256

A short example
===============
This is a short example of how to use *Meristem Phenotyper 3D*. Say, the image
has the name ''image.tif''. The following code fits a contour onto the image.

.. figure:: images/raw_image.png
   :height: 256
   :width: 256
   
   3D view of image.tif

.. code:: python

  import Meristem_Phenotyper_3D as ap
  
  image, _ = tiffread('image.tif')  # import the image as numpy array
  A = ap.AutoPhenotype(data = image)  # initialise object
  A.set_contour_to_box()  # initialise contour for fit
  A.contour_fit(weighting = 1., iterate_smooth = 1)  # fit

After this code is run, both the image data and the contour are stored in the
object A.

.. figure:: images/ACWE_animation-crop.gif
   :scale: 50 %
   
   ACWE fit onto a SAM. Visualisation of the code above.

In the next step, the contour is converted into a mesh, which is then smoothed
and sliced along its curvature to prepare for the extraction of the SAM
features.

.. code:: python
  
  A.mesh_conversion()  # create a mesh of the contours surface
  A.smooth_mesh(iterations = 200, relaxation_factor = 0.2)  # smoothe the mesh
  A.curvature_slice(threshold = 0., curvature_type = 'mean')  # slice the mesh

The mesh data is also stored in A.

Now that the mesh is sliced, the next step is to extract the SAMs features and
start the evaluation by fitting spheres onto the extracted features.

.. code:: python

  A.feature_extraction(1)  # extract the features of the meristem
  A.sphere_fit()  # fit spheres onto the extracted features
  A.sphere_evaluation()  # calculate the results of the sphere fit
  A.paraboloid_fit_mersitem()  # fits the meristem with a paraboloid

The features and results are stored in A.

The results can be accessed by using

.. code:: python

  print A.results  # display all results
  print A.get_div_angle(condition = 'sphere_R')  # display the divergence angle
                                                 # sorted by condition
  A.show_spheres()  # displays the fitted spheres in a 3D viewer

The object A can be saved and loaded at any point of the decribed process by 
using

.. code:: python

  A.save('/path/foldername')
  A.load('/path/foldername')

Results
=======
All length, area or volume results are in units of pixel of the original image.
Angles are in degree.
The results of the algoritm are described in the following:

points_in_feature: 
------------------
Number of data points in feature (either meristem or primordium). 

.. figure:: cartoons/points_in_feature.png
   :height: 256
   :width: 256
   
curvature:
----------
The type of curvature is the same as specified in the curvature_slice function.
 
* mean_curvature: Curvature in each point of the feature devided by the total 
  number of points.
* std_curvature: Standard deviation of the mean curvtaure.
* max_curvature: Maximum value of curvature in the feature.
* min_curvature: Minimum value of curvature in the feature.

.. figure:: cartoons/curvature.png
   :height: 256
   :width: 256

absolute coordinates:
---------------------
sphere_x_abs, sphere_y_abs, sphere_z_abs. Coordinates of the centers of the
fitted spheres in the coordinate system of the original image.

.. figure:: cartoons/abs_coords.png
   :height: 256
   :width: 256

sphere_radius:
--------------
Radius of the fitted sphere. From this the sphere_volume is computed.

.. figure:: cartoons/sphere_radius.png
   :height: 256
   :width: 256

sphere_res_var:
---------------
Residual variance of the sphere fit. 

.. figure:: cartoons/res_var.png
   :height: 256
   :width: 256

relative coordinates:
---------------------
sphere_x_rel, sphere_y_rel, sphere_z_rel. Coordinates of the centers of the
fitted spheres in a coordinate system, where the center of the meristem is the 
origin.

.. figure:: cartoons/relative_coords.png
   :height: 256
   :width: 256

sphere_angle_raw:
-----------------
Angles between the primordia as shown in the image.

.. figure:: cartoons/angles_raw.png
   :height: 256
   :width: 256

paraboloid fit:
---------------
* para_p1,..,p5: Parameters of the paraboloid fit.
* para_alpha, beta, gamma: Rotation of the paraboloid relative to the coordinate
  system of the original image. alpha: rotation around x axis, beta: axis, gamma: 
  z axis.
* para_apex_x,y,z: Absolute location of the paraboloids apex in the coordinate 
  system of the original image. 


.. figure:: cartoons/paraboloid.png
   :height: 256
   :width: 256

References
==========

.. [1] *A morphological approach to curvature-based evolution
   of curves and surfaces*. Pablo Márquez-Neila, Luis Baumela, Luis Álvarez.
   In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI). 
   2014
