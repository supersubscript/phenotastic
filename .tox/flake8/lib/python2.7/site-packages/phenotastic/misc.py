#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:00:11 2018

@author: henrik
"""

# -*- coding: utf-8 -*-
import pickle
import numpy as np
import time
import scipy.optimize as opt
import vtk
from vtk.util import numpy_support as nps
import Meristem_Phenotyper_3D as ap


def merge(lists):
    """ Merge lists based on overlapping elements.

    Parameters
    ----------
    lists : list of lists
        Lists to be merged based on overlap.

    Returns
    -------
    sets : list
        Minimal list of independent sets.

    """
    sets = [set(lst) for lst in lists if lst]
    merged = 1
    while merged:
        merged = 0
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = 1
                    common |= x
            results.append(common)
        sets = results
    return sets


def reject_outliers(data, n=2.):
    """ Remove outliers outside of @n standard deviations.

    Parameters
    ----------
    data : np.array
        1D array containing data to be filtered.

    n : float
         Number of standard deviations that should be included in final data.
         (Default value = 2.)

    Returns
    -------

    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < n]


def squared_dist_para(data, p, shiftx=0, shifty=0, shiftz=0, shiftcurv=0):
    """

    Parameters
    ----------
    data :

    p :

    shiftx :
         (Default value = 0)
    shifty :
         (Default value = 0)
    shiftz :
         (Default value = 0)
    shiftcurv :
         (Default value = 0)

    Returns
    -------

    """
    return np.sum((p[0] - shiftcurv) * (data[:, 0] - shiftx)**2
                  + (p[1] - shiftcurv) * (data[:, 1] - shifty)**2
                  + p[2] * (data[:, 0] - shiftx)
                  + p[3] * (data[:, 1] - shifty)
                  + p[4]
                  - data[:, 2])**2


def paraboloid(x, y, p):
    """ Return the z-value for a paraboloid given input xy-coordinates and
    parameters.

    Parameters
    ----------
    x : float

    y : float

    p : np.ndarray
        Paraboloid parameters.

    Returns
    -------

    """
    p1, p2, p3, p4, p5 = p
    return p1 * x**2 + p2 * y**2 + p3 * x + p4 * y + p5


def swaprows(a, how=[2, 0, 1]):
    """

    Parameters
    ----------
    a :

    how :
         (Default value = [2)
    0 :

    1] :


    Returns
    -------

    """
    a[:, [0, 1, 2]] = a[:, how]
    return a


def radius(x, y):
    """

    Parameters
    ----------
    x :

    y :


    Returns
    -------

    """
    return np.sqrt(x**2 + y**2)


def sort_columns(a):
    """

    Parameters
    ----------
    a :


    Returns
    -------

    """
    for i in range(np.shape(a)[0]):
        if a[i, 0] < a[i, 1]:
            a[i, [0, 1]] = a[i, [1, 0]]
    return a


def close_window(iren):
    """

    Parameters
    ----------
    iren :


    Returns
    -------

    """
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()


def render_four_viewports(actors, viewports):
    """

    Parameters
    ----------
    actors :

    viewports :


    Returns
    -------

    """
    assert(len(actors) == len(viewports))
    actors = np.array(actors)
    viewports = np.array(viewports)

    # Set viewport ranges
    xmins = [0.0, 0.5, 0.0, 0.5]
    xmaxs = [0.5, 1.0, 0.5, 1.0]
    ymins = [0.0, 0.0, 0.5, 0.5]
    ymaxs = [0.5, 0.5, 1.0, 1.0]

    rw = vtk.vtkRenderWindow()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)
    for ii in xrange(4):
        ren = vtk.vtkRenderer()
        rw.AddRenderer(ren)
        ren.SetViewport(xmins[ii], ymins[ii], xmaxs[ii], ymaxs[ii])

        for jj in [i for i, x in enumerate(viewports == ii) if x]:
            ren.AddActor(actors[jj])
            ren.ResetCamera()
        ren.ResetCamera()
    rw.Render()
    rw.SetWindowName('RW: Multiple ViewPorts')
    iren.Start()


def outlineactor(poly, opacity=.5):
    """ Return an outline actor for a given polydata.

    Parameters
    ----------
    poly : vtk.PolyData or vtkInterface.PolyData
        PolyData object to get outline actor for.

    Returns
    -------

    """
    outline = vtk.vtkOutlineFilter()
    outline.SetInput(poly.GetOutput())

    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInput(outline.GetOutput())

    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(outlineMapper)
    outlineActor.GetProperty().SetOpacity(opacity)
    return outlineActor


def tic(name='time1'):
    """ Elapsed time tracking function. See @toc.

    Parameters
    ----------
    name :
         (Default value = 'time1')

    Returns
    -------

    """
    globals()[name] = time.time()


def toc(name='time1', verbose=True):
    """ Elapsed time tracking function. See @tic.

    Parameters
    ----------
    name :
         (Default value = 'time1')
    print_it :
         (Default value = True)

    Returns
    -------

    """
    total_time = time.time() - globals()[name]
    if verbose == True:
        if total_time < 0.001:
            print '--- ', round(total_time * 1000., 2), 'ms', ' ---'
        elif total_time >= 0.001 and total_time < 60:
            print '--- ', round(total_time, 2), 's', ' ---'
        elif total_time >= 60 and total_time / 3600. < 1:
            print '--- ', round(total_time / 60., 2), 'min', ' ---'
        else:
            print '--- ', round(total_time / 3600., 2), 'h', ' ---'
    else:
        return total_time


def readImages(fname):
    """ Read images using the legacy tiffread function from TissueViewer. To be
    removed.

    Parameters
    ----------
    imageFileName :


    Returns
    -------

    """
    image, tags = tiffread(fname)
    return image, tags


def circle_levelset(shape, center, sqradius):
    """Build a binary function with a circle as the 0.5-levelset.

    Parameters
    ----------
    shape :

    center :

    sqradius :


    Returns
    -------

    """
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u


def fit_sphere(data, init=[0, 0, 0, 10]):
    """

    Parameters
    ----------
    data :

    init :
         (Default value = [0)
    0 :

    10] :


    Returns
    -------

    """
    def fitfunc(p, coords):
        """

        Parameters
        ----------
        p :

        coords :


        Returns
        -------

        """
        x0, y0, z0, _ = p
        x, y, z = coords.T
        return ((x - x0)**2 + (y - y0)**2 + (z - z0)**2)

    def errfunc(p, x): return fitfunc(p, x) - p[3]**2.
    p1, _ = opt.leastsq(errfunc, init, args=(np.array(np.nonzero(data)).T,))
    p1[3] = abs(p1[3])
    return p1


def view3d(data, contour=False):
    """

    Parameters
    ----------
    p :

    x :


    Returns
    -------

    """
    from mayavi import mlab
    data = np.array(data)
    if data.dtype == 'bool':
        data = np.array(data, dtype='int')
    mlab.gcf()
    mlab.clf()
    if contour == False:
        mlab.points3d(np.nonzero(data)[0], np.nonzero(data)[
                      1], np.nonzero(data)[2], scale_factor=.5)
    else:
        mlab.contour3d(data, contours=[0.5])
    mlab.show()


def save_var(variables, path, confirm=False):
    """

    Parameters
    ----------
    variables :

    path :

    confirm :
         (Default value = False)

    Returns
    -------

    """
    with open(path, 'w') as f:
        pickle.dump(variables, f)
    if confirm != False:
        print 'all saved'


def load_var(path):
    """

    Parameters
    ----------
    path :


    Returns
    -------

    """
    with open(path) as f:
        return pickle.load(f)


def shake(array):
    """

    Parameters
    ----------
    array :


    Returns
    -------

    """
    msk = np.array(array)
    msk[1::, :, :] = msk[:-1:, :, :] + msk[1::, :, :]
    msk[:-1:, :, :] = msk[:-1:, :, :] + msk[1::, :, :]
    msk[:, 1::, :] = msk[:, :-1:, :] + msk[:, 1::, :]
    msk[:, :-1:, :] = msk[:, :-1:, :] + msk[:, 1::, :]
    msk[:, :, 1::] = msk[:, :, 1:] + msk[:, :, :-1:]
    msk[:, :, :-1:] = msk[:, :, 1:] + msk[:, :, :-1:]
    return np.array(msk, dtype='bool')


def sort_a_along_b(b, a):
    """ Sort a along b.

    Parameters
    ----------
    b :

    a :


    Returns
    -------

    """
    return np.array(sorted(zip(a, b)))[:, 1]
#


def spherefit_results(spheres):
    """ Legacy method retrieving results from a sphere-fit operation.

    Gives several results from an array of spheres, such as distance between the first sphere (mersitem) and the other spheres (organs).
    Input:
        np.array[[x_center_meristem, y_center_meristem, z_center_meristem, radius_mersitem],
                [x_center_organ1, y_center_organ1, z_center_organ1, radius_organ1]
                ...]
    Output:
        np.array[[voulme_meristem, 0,0,0,0,0,0]
                [volume_organ1, location_organ1_realtive_to_meristem_x, y, z, r, theta, phi, projected_theta]
                ...]

    Note
    ----
    For spherical coordinates:  xyz -> yzx
    Angles in radians, distances in voxel

    Parameters
    ----------
    spheres : list of sphere PolyData


    Returns
    -------

    """

    num_obj = np.shape(spheres)[0]
    out = np.zeros((num_obj, 8))

    def sphere_voulume(radius):
        """

        Parameters
        ----------
        radius :


        Returns
        -------

        """
        return 4. / 3. * np.pi * radius**3.

    out[:, 0] = sphere_voulume(spheres[:, -1])  # voulumes
    out[1:, 1] = spheres[1:, 0] - spheres[0, 0]  # x relative to meristem
    out[1:, 2] = spheres[1:, 1] - spheres[0, 1]  # y
    out[1:, 3] = spheres[1:, 2] - spheres[0, 2]  # z
    out[1:, 4] = np.sqrt(out[1:, 1]**2. + out[1:, 2]**2. + out[1:, 3]**2.)  # r
    out[1:, 5] = np.arccos(out[1:, 1] / out[1:, 4])  # theta
    out[1:, 6] = np.arctan(out[1:, 3] / out[1:, 2])  # phi
    out[1:, 7] = np.arctan2(out[1:, 2], out[1:, 3])
    for i in range(1, num_obj):
        if out[i, 7] < 0:
            out[i, 7] = out[i, 7] + 2. * np.pi

    return out


def get_max_contrast_colours(n=64):
    """ Get colors with maximal inter-color contrast.

    Parameters
    ----------
    n : Numbers of RGB colors to return.
         (Default value = 64)

    Returns
    -------

    """
    rgbs = [[0, 0, 0],
            [1, 0, 103],
            [213, 255, 0],
            [255, 0, 86],
            [158, 0, 142],
            [14, 76, 161],
            [255, 229, 2],
            [0, 95, 57],
            [0, 255, 0],
            [149, 0, 58],
            [255, 147, 126],
            [164, 36, 0],
            [0, 21, 68],
            [145, 208, 203],
            [98, 14, 0],
            [107, 104, 130],
            [0, 0, 255],
            [0, 125, 181],
            [106, 130, 108],
            [0, 174, 126],
            [194, 140, 159],
            [190, 153, 112],
            [0, 143, 156],
            [95, 173, 78],
            [255, 0, 0],
            [255, 0, 246],
            [255, 2, 157],
            [104, 61, 59],
            [255, 116, 163],
            [150, 138, 232],
            [152, 255, 82],
            [167, 87, 64],
            [1, 255, 254],
            [255, 238, 232],
            [254, 137, 0],
            [189, 198, 255],
            [1, 208, 255],
            [187, 136, 0],
            [117, 68, 177],
            [165, 255, 210],
            [255, 166, 254],
            [119, 77, 0],
            [122, 71, 130],
            [38, 52, 0],
            [0, 71, 84],
            [67, 0, 44],
            [181, 0, 255],
            [255, 177, 103],
            [255, 219, 102],
            [144, 251, 146],
            [126, 45, 210],
            [189, 211, 147],
            [229, 111, 254],
            [222, 255, 116],
            [0, 255, 120],
            [0, 155, 255],
            [0, 100, 1],
            [0, 118, 255],
            [133, 169, 0],
            [0, 185, 23],
            [120, 130, 49],
            [0, 255, 198],
            [255, 110, 65],
            [232, 94, 190]]
    return rgbs[0:n]
