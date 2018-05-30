#from tissueviewer.tvtiff import tiffread
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import time
import scipy.optimize as opt
import vtk
from vtk.util import numpy_support as nps
import Meristem_Phenotyper_3D as ap


def reject_outliers_2(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]


def squared_dist_para(data, p, shiftx=0, shifty=0, shiftz=0, shiftcurv=0):
    return np.sum((p[0] - shiftcurv) * (data[:, 0] - shiftx)**2
                  + (p[1] - shiftcurv) * (data[:, 1] - shifty)**2
                  + p[2] * (data[:, 0] - shiftx)
                  + p[3] * (data[:, 1] - shifty)
                  + p[4]
                  - data[:, 2])**2


def paraboloid(x, y, p):
    p1, p2, p3, p4, p5 = p
    return p1 * x**2 + p2 * y**2 + p3 * x + p4 * y + p5


def swaprows(a, how=[2, 0, 1]):
    a[:, [0, 1, 2]] = a[:, how]
    return a


def radius(x, y):
    return np.sqrt(x**2 + y**2)


def sort_columns(a):
    for i in range(np.shape(a)[0]):
        if a[i, 0] < a[i, 1]:
            a[i, [0, 1]] = a[i, [1, 0]]
    return a


def close_window(iren):
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()


def render_four_viewports(actors, viewports):
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

def polydata_from_arrays(verts, faces):
  # Get points
  points = vtk.vtkPoints()
  points.SetData(nps.numpy_to_vtk(verts, array_type=vtk.VTK_FLOAT, deep=True))

  # Create polygons
  nFaces = len(faces)
  faces = np.array([np.append(len(ii), ii) for ii in faces]).flatten()
  polygons = vtk.vtkCellArray()
  polygons.SetCells(nFaces, nps.numpy_to_vtk(faces, array_type=vtk.VTK_ID_TYPE))

  # Create polydata from points and polygons
  polygonPolyData = vtk.vtkPolyData()
  polygonPolyData.SetPoints(points)
  polygonPolyData.SetPolys(polygons)
  polygonPolyData.Update()
  return polygonPolyData

def render_actors(actors, colorbar=False, ruler=False, bg = [.1,.2,.3]):
    # Setup the window
    ren1 = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    for ii in actors:
        ren1.AddActor(ii)
    ren1.SetBackground(bg[0], bg[1], bg[2])

    if colorbar:
      scalarBar = actor_to_colorbar(actors[0])
      ren1.AddActor2D(scalarBar)

    distanceWidget = vtk.vtkDistanceWidget()
    distanceWidget.SetInteractor(iren)
    distanceWidget.CreateDefaultRepresentation()
#    static_cast<vtkDistanceRepresentation *>(distanceWidget->GetRepresentation())->SetLabelFormat("%-#6.3g mm");

    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(iren)
    widget.SetViewport(0.0, 0.0, 0.1, 0.1)
    widget.SetEnabled(1)
    widget.InteractiveOn()
    if ruler:
      distanceWidget.On()
    else:
      distanceWidget.Off()
    ren1.ResetCamera()

    # Render and interact
    renWin.Render()
    iren.Initialize()
    iren.Start()

    close_window(iren)
    del renWin, iren


def return_actor(obj, col='red'):
    # TODO: Colours wrong
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(obj.GetOutputPort())
    if col == 'red':
        contour.GenerateValues(1, 0, 0)
    elif col == 'green':
        contour.GenerateValues(0, 1, 0)
    elif col == 'blue':
        contour.GenerateValues(0, 0, 1)

    # map the contours to graphical primitives
    contourMapper = vtk.vtkPolyDataMapper()
    contourMapper.SetInput(contour.GetOutput())
    contourMapper.SetScalarRange(0.0, 1.2)

    contourActor = vtk.vtkActor()
    contourActor.SetMapper(contourMapper)
    contourActor.GetProperty().SetOpacity(0.5)

    return contourActor

def actor_to_colorbar(actor):
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(actor.GetMapper().GetLookupTable())
#    arrayName = actor.GetMapper().GetInput().GetPointData().GetScalars().GetName()
    scalarBar.SetTitle("")
    scalarBar.SetNumberOfLabels(4)

    # Create a lookup table to share between the mapper and the scalarbar
#    hueLut = vtk.vtkLookupTable()
#    hueLut.DeepCopy(actor.GetMapper().GetLookupTable())
#    hueLut.Build()

    scalarBar.SetLookupTable(actor.GetMapper().GetLookupTable())
    return scalarBar


def return_outlineActor(obj):
    outline = vtk.vtkOutlineFilter()
    outline.SetInput(obj.GetOutput())

    # map it to graphics primitives
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInput(outline.GetOutput())

    # create an actor for it
    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(outlineMapper)
    outlineActor.GetProperty().SetOpacity(0.5)
    return outlineActor


def polyData_to_cart(polyData, dim=3):
    geomPoints = np.zeros((polyData.GetNumberOfPoints(), dim))
    for ii in xrange(0, polyData.GetNumberOfPoints()):
        geomPoints[ii] = polyData.GetPoint(ii)
    return geomPoints


def array_to_polyData(data):
    polydata = vtk.vtkPolyData()
    vtkArray = nps.numpy_to_vtk(
        num_array=data, deep=True, array_type=vtk.VTK_FLOAT)
    vtkPoints = vtk.vtkPoints()
    vtkPoints .SetData(vtkArray)
    polydata.SetPoints(vtkPoints)
    return(polydata)


def tic(name='time1'):
    globals()[name] = time.time()


def toc(name='time1', print_it=True):
    total_time = time.time() - globals()[name]
    if print_it == True:
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


def readImages(imageFileName):
    image, tags = tiffread(imageFileName)
    return image, tags


def circle_levelset(shape, center, sqradius):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u


def fit_sphere(data, init=[0, 0, 0, 10]):
    def fitfunc(p, coords):
        x0, y0, z0, _ = p
        x, y, z = coords.T
        return ((x - x0)**2 + (y - y0)**2 + (z - z0)**2)

    def errfunc(p, x): return fitfunc(p, x) - p[3]**2.
    p1, _ = opt.leastsq(errfunc, init, args=(np.array(np.nonzero(data)).T,))
    p1[3] = abs(p1[3])
    return p1


def view3d(data, contour=False):
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
    with open(path, 'w') as f:
        pickle.dump(variables, f)
    if confirm != False:
        print 'all saved'


def load_var(path):
    with open(path) as f:
        return pickle.load(f)


def shake(array):
    msk = np.array(array)
    msk[1::, :, :] = msk[:-1:, :, :] + msk[1::, :, :]
    msk[:-1:, :, :] = msk[:-1:, :, :] + msk[1::, :, :]
    msk[:, 1::, :] = msk[:, :-1:, :] + msk[:, 1::, :]
    msk[:, :-1:, :] = msk[:, :-1:, :] + msk[:, 1::, :]
    msk[:, :, 1::] = msk[:, :, 1:] + msk[:, :, :-1:]
    msk[:, :, :-1:] = msk[:, :, 1:] + msk[:, :, :-1:]
    return np.array(msk, dtype='bool')


def sort_a_along_b(b, a):
    return np.array(sorted(zip(a, b)))[:, 1]
#


def view_polydata(poly):
    if np.shape(poly) == ():
        numel = 1
        poly = [poly]
    else:
        numel = np.shape(poly)[0]
    colors = [(1., 1., 1.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1., 1., 0.),
              (1., 0., 1.), (0., 1., 1.), (1., 0.5,
                                           0.5), (0.5, 1., 0.5), (0.5, 0.5, 1.),
              (1., 1., 0.5), (1., 0.5, 1.), (0.5, 1., 1.)]  # this is not very nice
    Mappers = []
    Actors = []
    render = vtk.vtkRenderer()
    for i in range(numel):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(poly[i])
        mapper.ScalarVisibilityOff()
        mapper.Update()
        Mappers.append(mapper)
        actor = vtk.vtkActor()
        actor.SetMapper(Mappers[i])
        actor.GetProperty().SetColor(colors[i])
        Actors.append(actor)
        render.AddActor(Actors[i])
    renderwindow = vtk.vtkRenderWindow()
    renderwindow.AddRenderer(render)
    renderwindow.SetSize(600, 600)
    interactrender = vtk.vtkRenderWindowInteractor()
    interactrender.SetRenderWindow(renderwindow)
    interactrender.Initialize()
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactrender)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    widget.SetEnabled(1)
    widget.InteractiveOn()
    render.ResetCamera()
    renderwindow.Render()
    interactrender.Start()

def set_to_numpy(set_):
    return np.array(list(set_))

# def array_from_vtk_polydata(poly,size=[]):
#     if np.shape(size) == np.shape([]):
#         size = np.array(poly.GetPoints().GetBounds(),dtype='int')[1::2]
#     indices = np.array(vtk_to_numpy(poly.GetPoints().GetData()),dtype='int')
#     out = np.zeros(size)
#     out[indices[:,0]-1,indices[:,1]-1,indices[:,2]-1] = 1
#     return np.array(out)


# def vtk_polydata_from_array(array):
#     out = vtk.vtkPolyData()
#     longarray = (numpy_to_vtk(np.nonzero(array)))
#     out = vtk.vtkPointData()
#     out.SetInputData(longarray)
#     NumPy_data_shape = array.shape
#     VTK_data = numpy_to_vtk(num_array=array.ravel(), deep=True, array_type=vtk.VTK_POINT_DATA)


def spherefit_results(spheres):
    """
    gives several results from an array of spheres, such as distance between the first sphere (mersitem) and the other spheres (organs).
    Input:
        np.array[[x_center_meristem, y_center_meristem, z_center_meristem, radius_mersitem],
                [x_center_organ1, y_center_organ1, z_center_organ1, radius_organ1]
                ...]
    Output:
        np.array[[voulme_meristem, 0,0,0,0,0,0]
                [volume_organ1, location_organ1_realtive_to_meristem_x, y, z, r, theta, phi, projected_theta]
                ...]
    note: for spherical coordinates:  xyz -> yzx
    angels in rad, distances in voxel
    """

    num_obj = np.shape(spheres)[0]
    out = np.zeros((num_obj, 8))

    def sphere_voulume(radius):
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


def visualise_paraboloid_and_mesh(A):
    results = A.results
    meristem = results.filter(like='para_p').iloc[0]
    alpha, beta, gamma = results.filter(['para_alpha', 'para_beta',
                                         'para_gamma']).iloc[0]
    p1, p2, p3, p4, p5 = meristem
    # Configure
    quadric = vtk.vtkQuadric()
    quadric.SetCoefficients(p1, p2, 0, 0, 0, 0, p3, p4, -1, p5)
    # Quadric.SetCoefficients(p1, p2, 0, 0, 0, 0, p3, p4, -1, p5)
    # F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2 + a3*x*y + a4*y*z + a5*x*z + a6*x + a7*y

    # + a8*z + a9
    sample = vtk.vtkSampleFunction()
    sample.SetSampleDimensions(100, 100, 100)
    sample.SetImplicitFunction(quadric)
    upperBound = 2000
    lowerBound = -2000
    sample.SetModelBounds([lowerBound, upperBound] * 3)

    # Create the paraboloid contour
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(sample.GetOutputPort())
    contour.GenerateValues(1, 1, 1)
    contour.Update()
    contourMapper = vtk.vtkPolyDataMapper()
    contourMapper.SetInput(contour.GetOutput())
    contourMapper.SetScalarRange(0.0, 1.2)
    contourActor = vtk.vtkActor()
    contourActor.SetMapper(contourMapper)
    contourActor.GetProperty().SetOpacity(0.1)

    rotMat = ap.rot_matrix_44([alpha, beta, gamma], invert=True)
    trans = vtk.vtkMatrix4x4()
    for ii in xrange(0, rotMat.shape[0]):
        for jj in xrange(0, rotMat.shape[1]):
            trans.SetElement(ii, jj, rotMat[ii][jj])

    transMat = vtk.vtkMatrixToHomogeneousTransform()
    transMat.SetInput(trans)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(contour.GetOutputPort())
    transformFilter.SetTransform(transMat)
    transformFilter.Update()

    transformedMapper = vtk.vtkPolyDataMapper()
    transformedMapper.SetInputConnection(transformFilter.GetOutputPort())
    transformedActor = vtk.vtkActor()
    transformedActor.SetMapper(transformedMapper)
    transformedActor.GetProperty().SetColor(0, 1, 0)
    transformedActor.GetProperty().SetOpacity(0.2)

    # Input sphere with top coordinates for paraboloid (corrected)
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(results['para_apex_x'][0], results[
                           'para_apex_y'][0], results['para_apex_z'][0])
    sphereSource.SetRadius(1)
    sphereSource.Update()
    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)

    # Input the mesh
    meshData = A.mesh
    meshMapper = vtk.vtkPolyDataMapper()
    meshMapper.SetInput(meshData)
    meshActor = vtk.vtkActor()
    meshActor.SetMapper(meshMapper)
    meshActor.GetProperty().SetOpacity(0.2)

    ''' Plotting '''
    # Setup the window
    ren1 = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    ren1.AddActor(meshActor)
    ren1.AddActor(sphereActor)
    ren1.AddActor(transformedActor)
    ren1.SetBackground(.5, .5, .5)  # Background color white

    # Render and interact
    renWin.Render()
    iren.Start()
    close_window(iren)
    del renWin, iren


def get_paraboloid_axes_actor(length=10, origin=[0, 0, 0], p0=[0, 0, 0], p1=[10, 0, 0], rotMat=None):
    linesPolyData = vtk.vtkPolyData()

    # Create a vtkPoints container and store the points in it
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(origin)
    pts.InsertNextPoint(p0)
    pts.InsertNextPoint(p1)

    # Add the points to the polydata container
    linesPolyData.SetPoints(pts)

    # Create the first line (between Origin and P0)
    line0 = vtk.vtkLine()

    # the second 0 is the index of the Origin in linesPolyData's points
    line0.GetPointIds().SetId(0, 0)

    # the second 1 is the index of P0 in linesPolyData's points
    line0.GetPointIds().SetId(1, 1)

    # Create the second line (between Origin and P1)
    line1 = vtk.vtkLine()
    # the second 0 is the index of the Origin in linesPolyData's points
    line1.GetPointIds().SetId(0, 0)
    line1.GetPointIds().SetId(1, 2)  # 2 is the index of P1 in linesPolyData's points

    # Create a vtkCellArray container and store the lines in it
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(line0)
    lines.InsertNextCell(line1)

    # Add the lines to the polydata container
    linesPolyData.SetLines(lines)

    # Create two colors - one for each line
    red = [255, 0, 0]
    green = [0, 255, 0]

    # Create a vtkUnsignedCharArray container and store the colors in it
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.InsertNextTupleValue(red)
    colors.InsertNextTupleValue(green)

    # Color the lines.
    # SetScalars() automatically associates the values in the data array passed as parameter
    # to the elements in the same indices of the cell data array on which it is called.
    # This means the first component (red) of the colors array
    # is matched with the first component of the cell array (line 0)
    # and the second component (green) of the colors array
    # is matched with the second component of the cell array (line 1)
    linesPolyData.GetCellData().SetScalars(colors)

    # Setup the visualisation pipeline
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(linesPolyData)

    actor = vtk.vtkActor()
    actor = actor.SetMapper(mapper)

    return(actor)


def rotate_agent(polyData, alpha, beta, gamma, invert=False):
    rotMat = ap.rot_matrix_44([alpha, beta, gamma], invert=invert)
    trans = vtk.vtkMatrix4x4()
    for ii in xrange(0, rotMat.shape[0]):
        for jj in xrange(0, rotMat.shape[1]):
            trans.SetElement(ii, jj, rotMat[ii][jj])

    transMat = vtk.vtkMatrixToHomogeneousTransform()
    transMat.SetInput(trans)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(polyData.GetOutputPort())
    transformFilter.SetTransform(transMat)
    transformFilter.Update()

    transformedMapper = vtk.vtkPolyDataMapper()
    transformedMapper.SetInputConnection(transformFilter.GetOutputPort())
    transformedActor = vtk.vtkActor()
    transformedActor.SetMapper(transformedMapper)
    return(transformedActor)


def merge_lists_to_sets(lsts):
    sets = [set(lst) for lst in lsts]
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


def get_max_contrast_colours(n=64):
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
