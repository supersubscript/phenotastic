
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:01:39 2018

@author: henrik
"""
import vtk
import handy_functions as hf

def new_smooth(A, iters=5, featureAngle=45, passBand=0.001):
  smoother = vtk.vtkWindowedSincPolyDataFilter()
  smoother.SetInputConnection(A.mesh.GetProducerPort())
  smoother.SetNumberOfIterations(iters)
  smoother.BoundarySmoothingOff()
  smoother.FeatureEdgeSmoothingOff()
  smoother.SetFeatureAngle(featureAngle)
  smoother.SetPassBand(passBand)
  smoother.NonManifoldSmoothingOn()
  smoother.NormalizeCoordinatesOn()
  smoother.Update()

  smoothedMapper = vtk.vtkPolyDataMapper()
  smoothedMapper.SetInputConnection(smoother.GetOutputPort())
  return smoothedMapper.GetInput()

def plot_boundary(A):
  featureEdges = vtk.vtkFeatureEdges()
  featureEdges.SetInputConnection(A.mesh.GetProducerPort())
  featureEdges.BoundaryEdgesOn()
  featureEdges.FeatureEdgesOff()
  featureEdges.ManifoldEdgesOff()
  featureEdges.NonManifoldEdgesOff()
  featureEdges.Update()

  edgeMapper = vtk.vtkPolyDataMapper()
  edgeMapper.SetInputConnection(featureEdges.GetOutputPort())
  edgeActor = vtk.vtkActor()
  edgeActor.SetMapper(edgeMapper)

  diskMapper = vtk.vtkPolyDataMapper()
  newPoly = vtk.vtkPolyData()
  newPoly.ShallowCopy(A.mesh)
  diskMapper.SetInputConnection(newPoly.GetProducerPort())
  diskActor = vtk.vtkActor()
  diskActor.SetMapper(diskMapper)

  hf.render_actors([edgeActor, diskActor])

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

def swaprows(a, how=[2, 0, 1]):
    """

    Parameters
    ----------
    a :

    how :
         (Default value = [2,0,1] :


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

def remove_tongues_naive(A, radius, threshold, holefill=100):
    """

    Parameters
    ----------
    A :

    radius :

    threshold :

    holefill :
         (Default value = 100)

    Returns
    -------

    """
    from scipy.spatial import KDTree
    mesh = A.mesh

    # Preprocessing
    mesh = remove_bridges(A)
    mesh.ExtractLargest()
    mesh.Clean()
    mesh.FillHoles(holefill)

    boundary = mesh.ExtractEdges(boundary_edges=True, non_manifold_edges=False,
                                 feature_edges=False, manifold_edges=False)

    # Get points and corresponding indices
    bdpts = boundary.points
    pts = A.mesh.points
    mesh_idxs = np.array([mesh.FindPoint(ii) for ii in bdpts])

    # Get the neighbours and then the number of adjacent boundary indices
    tree = KDTree(bdpts)
    neighs = tree.query_ball_point(pts, radius)
    nneighs = np.array([len([jj for jj in ii if jj in mesh_idxs]) for ii in neighs])

    to_remove = nneighs > threshold
    A.mesh = A.mesh.RemovePoints(to_remove, keepscalars=False)[0]

    # Postprocessing
    mesh = remove_bridges(A)
    mesh = correct_bad_mesh(A.mesh)
    mesh.ExtractLargest()
    mesh.Clean()
    mesh.FillHoles(holefill)
    return mesh



    def show_curvatures(self, curv_types=['mean'], operations=[],
                        stdevs=2, numColors=2, curvs=None, normalise=False,
                        log_=False, return_actors = False):
        """

        Parameters
        ----------
        curv_types :
             (Default value = ['mean'])

        operations :
             (Default value = [])

        stdevs :
             (Default value = 2)

        numColors :
             (Default value = 2)

        curvs :
             (Default value = None)

        normalise :
             (Default value = False)

        return_actors :
             (Default value = False)

        Returns
        -------

        """
#        self = A
#        curv_types=['mean']
#        operations=[]
#        stdevs=2
#        numColors=2
#        curvs=curvs
#        normalise=False
#        log_=False
#        return_actors = False

        # TODO: This function needs a major rewrite
        output = vtk.vtkPolyData()
        output.ShallowCopy(self.mesh)
        polyAlg = self.mesh.GetProducerPort()

        curvaturesFilter = vtk.vtkCurvatures()
        curvaturesFilter.SetInputConnection(polyAlg)

        # Get curvature values
        if not isinstance(curvs, pd.DataFrame):
            self.calculate_curvatures(curv_types=curv_types, operations=operations)
            curvVals = self.mesh.GetPointData().GetArray(self.curvature_type)
            curvVals = pd.DataFrame(nps.vtk_to_numpy(curvVals))
            curvs = copy.deepcopy(curvs)
        else:
            curvVals = copy.deepcopy(curvs)

        if stdevs != "all":
            curvVals = misc.reject_outliers_2(curvVals, m=stdevs)

        if normalise:
          min_ = curvs.min()
          max_ = curvs.max()
          curvVals = (curvVals - min_) / (max_ - min_)

        scalarRange = [curvVals.min().values[0], curvVals.max().values[0]]

        vtkarr = nps.numpy_to_vtk(
                num_array=curvVals.values.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        vtkarr.SetName(self.curvature_type)
        curvaturesFilter.GetOutput().GetPointData().AddArray(vtkarr)
        curvaturesFilter.GetOutput().GetPointData().SetActiveScalars(self.curvature_type)

        # Create the color map
        colorLookupTable = vtk.vtkLookupTable()
        colorLookupTable.SetTableRange(scalarRange[0], scalarRange[1])
#        colorLookupTable.SetNanColor(255, 255, 255, 0.0) # Set Nan-color to black
        if log_:
          colorLookupTable.SetScaleToLog10()
        colorLookupTable.Build()

        # Generate the colors for each point based on the color map
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        colors.SetLookupTable(colorLookupTable)
        m = curvs.median()[0]

        for ii in xrange(self.mesh.GetNumberOfPoints()):
            val = curvVals.iloc[ii, 0]

            # Color nan values black or white depending on whether they are too big or too small
            dcolor = np.zeros(3)
            if np.isnan(val):
              if curvs.iloc[ii, 0] > m:
                dcolor = np.array([0.,0.,0.])
              elif curvs.iloc[ii, 0] < m:
                dcolor = np.array([1.,1.,1.])
            else:
              colorLookupTable.GetColor(val, dcolor)

            color = np.zeros(3, dtype='int16')
            for jj in xrange(3):
                color[jj] = 255 * dcolor[jj] / 1.0

            colors.InsertNextTupleValue(color)

        output.GetPointData().AddArray(colors)
        output.GetPointData().SetActiveScalars("Colors")

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(output.GetProducerPort())
        mapper.SetLookupTable(colorLookupTable)

        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # TODO: Add colorbar as well
        if return_actors:
          return actor

        misc.render_actors([actor], colorbar=True)
    def show_mesh(self, opacity=1.0):
        """
        Visualise the mesh.

        Uses vtk to visualise the mesh.
        This can be done before or after the segmentation (using
        self.curvatures_slice()).
        Note: This makes the script pause at the position of the call. Closing
        the render window lets the script continue.

        Parameters
        ----------
        opacity :
             (Default value = 1.0)

        Returns
        -------
        No return :
            Opens a render window.

        """
        view_polydata(self.mesh, (1., 1., 1.), (), opacity=opacity)

    def show_features(self):
        """
        Uses vtk to visualise the mesh.

        This can be done before or after the segmentation (using
        self.curvatures_slice() ).
        Note: This makes the script pause at the position of the call. Closing
        the render window lets the script continue.

        Parameters
        ----------

        Returns
        -------
        No return :
            Opens a render window.

        """
        view_polydata(self.features, (1., 1., 1.), ())

def create_logfile(logs, path=None):
    """
    Create a logfile for saving and loading AutoPhenotype data from boolean
    list.

    Return logfile and optionally save it as .csv file.

    Parameters
    ----------
    logs :

    path :
         (Default value = None)

    Returns
    -------


    """
    logfile = pd.DataFrame([logs], columns=['data',
                                            'contour',
                                            'mesh',
                                            'features',
                                            'results',
                                            'tags'])
    if type(path) == type(None):
        return logfile
    if type(path) == type('string'):
        logfile.to_csv(path)
        return logfile


def logfile_from_str(what, path=None):
    """Create a logfile for saving and loading AutoPhenotype data from keywords

    Uses keywords to generate a logfile. Keywords can be:
        *'all': everything below
        *'data': input data as .tif
        *'contour': contour fit as .tif
        *'mesh': mesh as vtk data .vtp
        *'features': features as vtk data .vtp
        *'results': results as .csv

    Parameters
    ----------
    what :

    path :
         (Default value = None)

    Returns
    -------


    """
    logs = [0, 0, 0, 0, 0, 0]
    if any(t == 'all' for t in what):
        logs = [1, 1, 1, 1, 1, 1]
    if any(t == 'data' for t in what):
        logs[0] = 1
    if any(t == 'contour' for t in what):
        logs[1] = 1
    if any(t == 'mesh' for t in what):
        logs[2] = 1
    if any(t == 'features' for t in what):
        logs[3] = 1
    if any(t == 'results' for t in what):
        logs[4] = 1
    if any(t == 'tags' for t in what):
        logs[5] = 1
    return create_logfile(logs, path)

def view_polydata(poly, firstcolor=(), lastcolor=(), return_actors=False, opacity=1.0, bg=(0, 0, 0)):
    """
    Display vtkPolyData. Can show superposition of many vtkPolyData.

    If input is a list of vtkPolyData, displays all of them in one viewer.

    Parameters
    ----------
    poly :

    firstcolor :

    Returns
    -------


    """
    if np.shape(poly) == ():
        numel = 1
        poly = [poly]
    else:
        numel = np.shape(poly)[0]
    if np.shape(firstcolor) != np.shape(()) and np.shape(
            lastcolor) != np.shape(()):
        colors = rgb_list(numel, firstcolor=firstcolor, lastcolor=lastcolor)
    elif np.shape(firstcolor) != np.shape(()):
        colors = rgb_list(numel, firstcolor=firstcolor)
    elif np.shape(lastcolor) != np.shape(()):
        colors = rgb_list(numel, lastcolor=lastcolor)
    else:
        colors = rgb_list(numel)
    Mappers = []
    Actors = []
    render = vtk.vtkRenderer()
    for i in range(numel):
        mapper = vtk.vtkPolyDataMapper()
        if int(vtk.vtkVersion().GetVTKVersion()[0]) < 6:
            mapper.SetInput(poly[i])
        else:
            mapper.SetInputData(poly[i])
        mapper.ScalarVisibilityOff()
        mapper.Update()
        Mappers.append(mapper)
        actor = vtk.vtkActor()
        actor.SetMapper(Mappers[i])
        actor.GetProperty().SetColor(colors[i])
        actor.GetProperty().SetOpacity(opacity)
        Actors.append(actor)
        render.AddActor(Actors[i])
    if return_actors == False:
        render.SetBackground(bg)
        renderwindow = vtk.vtkRenderWindow()
        renderwindow.AddRenderer(render)
        renderwindow.SetSize(1200, 1200)
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
    elif return_actors == True:
        return Actors


    def load2(self, where, what=None):
        """
        Load data to existing AutoPhenotype Object.

        What is loaded can be specified.

        Parameters
        ----------
        where :

        what :
             (Default value = None)

        Returns
        -------


        """
        if type(what) == type(None):
            logfile = pd.read_csv(where + '/logfile.csv')
        elif type(what) == type(create_logfile([1, 1, 1, 1, 1, 1])):
            logfile = what
        elif type(what) == type(['1', '2']):
            logfile = logfile_from_str(what)
        else:
            print ' - parameter what is unknown - '
            print 'If logfile.csv exists in directory (where) use what=None'
            print 'Else: either create logfile with create_logfile() function'
            print 'or specify files to load with strings:'
            print 'all, data, contour, mesh, features, results, tags'
            raise ValueError('parameter what is unknown')
        if logfile['data'][0] != 0:
            tiff = TIFF.open(where + "/data.tif", mode='r')
            ar = tiff.read_image()
            tiff.close()
            self.data = ar
#            self.data, _ = tiffread(where + '/data.tif')
        if logfile['contour'][0] != 0:
            tiff = TIFF.open(where + "/contour.tif", mode='r')
            ar = tiff.read_image()
            tiff.close()
            self.contour = ar
#            self.contour, _ = tiffread(where + '/contour.tif')
        if logfile['mesh'][0] != 0:
            meshreader = vtk.vtkXMLPolyDataReader()
            meshreader.SetFileName(where + '/mesh.vtp')
            meshreader.Update()
            self.mesh = PolyData(meshreader.GetOutput())
        if logfile['features'][0] != 0:
            self.features = []
            number_of_features = len(next(os.walk(where + '/features'))[2])
            for i in range(number_of_features):
                featurereader = vtk.vtkXMLPolyDataReader()
                featurereader.SetFileName(where + '/features/feature%s.vtp'
                                          % str(i))
                featurereader.Update()
                self.features.append(vtk.vtkPolyData())
                self.features[-1].DeepCopy(featurereader.GetOutput())
        if logfile['results'][0] != 0:
            self.results = pd.read_csv(where + '/results.csv')

    def load(self, where, what=None):
        """
        Load data to existing AutoPhenotype Object.

        What is loaded can be specified.

        Parameters
        ----------
        where :

        what :
             (Default value = None)

        Returns
        -------


        """
        if type(what) == type(None):
            logfile = pd.read_csv(where + '/logfile.csv')
        elif type(what) == type(create_logfile([1, 1, 1, 1, 1, 1])):
            logfile = what
        elif type(what) == type(['1', '2']):
            logfile = logfile_from_str(what)
        else:
            print ' - parameter what is unknown - '
            print 'If logfile.csv exists in directory (where) use what=None'
            print 'Else: either create logfile with create_logfile() function'
            print 'or specify files to load with strings:'
            print 'all, data, contour, mesh, features, results, tags'
            raise ValueError('parameter what is unknown')
        if logfile['data'][0] != 0:
            self.data, _ = tiffread(where + '/data.tif')
        if logfile['contour'][0] != 0:
            self.contour, _ = tiffread(where + '/contour.tif')
        if logfile['mesh'][0] != 0:
            meshreader = vtk.vtkXMLPolyDataReader()
            meshreader.SetFileName(where + '/mesh.vtp')
            meshreader.Update()
            self.mesh = PolyData(meshreader.GetOutput())
        if logfile['features'][0] != 0:
            self.features = []
            number_of_features = len(next(os.walk(where + '/features'))[2])
            for i in range(number_of_features):
                featurereader = vtk.vtkXMLPolyDataReader()
                featurereader.SetFileName(where + '/features/feature%s.vtp'
                                          % str(i))
                featurereader.Update()
                self.features.append(vtk.vtkPolyData())
                self.features[-1].DeepCopy(PolyData(featurereader.GetOutput()))
        if logfile['results'][0] != 0:
            self.results = pd.read_csv(where + '/results.csv')

