#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:56:10 2018

@author: henrik
"""

import numpy as np
from tifffile import TiffFile, imsave, tifffile

class tvfile(TiffFile):
    ''' Namespace storing the corresponding file stream and attributes governing
    (array) data and metadata.

    Attributes
    ----------
    fname : str, optional
        Path to the input file when loading.
    data : numpy.ndarray, optional
        File (intensity) data. Stored in ZCYX order by default.
    metadata : dict, optional
        File metadata.

    Note
    ----
    tvfile.asarray() and tvfile.data are not guaranteed to give the same result,
    as tvfile.asarray() reads the input file path for data.

    '''

    def __init__(self, fname='', data=None, metadata=None):
        if fname != '':
            super(tvfile, self).__init__(fname)
        self.data = data
        self.metadata = metadata

def tiffsave(fname, data, metadata=None, resolution=[1., 1., 1.], dtype=None,
             unit='um', segmented=False, backgroundlabel='', walllabel='', ncells='',
             segmentation_software='', segmentation_method='', parameters='',
             preprocessing='', extra_tags={},  **kwargs):
    ''' Save a file in ImageJ-compatible Tiff format.

    Parameters
    ----------
    fname : str
        Output file name.
    data : np.ndarray
        Array containing the image data.
    metadata : dict, optional
        Dictionary containing the metadata relevant to the file. Is saved as
        imagej format in the ImageJ 'ImageDescription' tag. If not given, will
        generate a default dictionary containing the required tags. Will be
        overwritten by other input arguments if specified, except for resolution.
        (Default = See text.)
    resolution : np.ndarray, optional
        Image resolution in ZYX order. (Default = [1.,1.,1.])
    dtype : np.dtype, optional
        dtype of the input array. Defaults to input array dtype unless
        specified. (Default = See text.)
    segmented : bool, optional
        Flag for whether the image is segmented or not. Needs to be set for
        certain other metadata flags. (Default = False)
    unit : str, optional
        Resolution unit. (Default = 'um')
    backgroundlabel : int, optional
        Label of the background in segmented image.
    walllabel : int, optional
        Label of the walls in segmented image.
    ncells : int, optional
        Number of cells in segmented image
    segmentation_software : str, optional
        Segmentation software used.
    segmentation_method : str, optional
        Segmentation method used.
    parameters : str, optional
        Segmentation parameters used.
    preprocessing : str, optional
        Preprocessing parameters/methods used.
    kwargs : dict, optional
        Extra arguments to tifffile.imsave.

    Note
    ----
    The metadata dict has priority over the otherwise specified tags.
    Resolution is set in order ZYX.

    '''

    metadata = {} if not metadata else metadata.copy()
    metadata.setdefault('spacing', resolution[0])
    metadata.setdefault('unit', unit)
    metadata.setdefault('slices', data.shape[0])
    metadata['segmented'] = segmented

    if metadata['segmented']:
        metadata['backgroundlabel'] = segmented
        metadata['walllabel'] = walllabel
        metadata['segmentation_software'] = segmentation_software
        metadata['segmentation_method'] = segmentation_method
        metadata['parameters'] = parameters
        metadata['preprocessing'] = preprocessing

    for ii in extra_tags:
        metadata[ii] = extra_tags[ii]

    imsave(fname, data=data, dtype=dtype,
           resolution=resolution[-1:-3:-1], imagej=True, metadata=metadata,
           **kwargs)

def tiffload(fname):
    ''' Loads an image file as a tifffile.TiffFile object.

    Parameters
    ----------
    fname : str
        Path to the file.
    '''
    fobj = tvfile(fname) #TiffFile(fname)

    def lowerdict(d):
        ''' Make all dict entries lowercase '''
        new_dict = dict()
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(d[k], dict):
                    new_dict[k.lower()] = lowerdict(d[k])
                elif isinstance(d[k], list):
                    new_dict[k.lower()] = [lowerdict(ii) for ii in d[k]]
                else:
                    new_dict[k.lower()] = v
        else:
            new_dict = d
        return new_dict

    metadata = dict()
    for ii in dir(fobj):
        if '_metadata' in ii and eval('fobj.' + ii):
            thisdict = lowerdict(eval('fobj.' + ii))
            for jj in thisdict:
                metadata[jj] = thisdict[jj]
    fobj.imagej_metadata = metadata

    pages = fobj.pages.pages
    ss = fobj.series[0]

    npages = len(fobj.pages.pages)
    nseries = len(fobj.series)

    cindex=ss.axes.find('C')
    if cindex > -1:
        nchannels = ss.shape[cindex]
    else:
        nchannels = 1

    # Reduce the series to the part of the data we actually want.
    newSeries = tifffile.TiffPageSeries(
        pages[::nseries], (npages / nseries, nchannels, pages[0].imagelength,
                           pages[0].imagewidth), dtype=ss.dtype, axes='ZCYX',
        stype='ImageJ')
    newSeries.offset = ss.offset
    newSeries.parent = ss.parent
    fobj.series = [newSeries]

    # Set pages according to new series
    fobj.pages.clear(fully=True)
    fobj.pages.pages = newSeries._pages

    # Assign tvfile attributes the new data
    fobj.data = fobj.asarray()
    fobj.metadata = fobj.imagej_metadata

    return fobj

def get_array(fobj, channel=-1):
    ''' Return the array of a TiffFile. Will be in Z(C)YX order. A negative
    channel value will return all channels. '''

    arr = fobj.asarray()
    arr = tifffile.tifffile.transpose_axes(arr, fobj.series[0].axes, 'TZCYXS')
    if channel > -1:
        arr = arr[0, :, channel, : ,:, 0]
    else:
        arr = arr[0, :, :, : ,:, 0]
    return arr

def get_resolution(file_):
    ''' Returns resolution of the image in ZYX order.

    Parameters
    ----------
    file_ : tifffile.TiffFile

    Note
    ----
    The z-resolution must have been stored as imagej metadata under the name
    'spacing'. Similarly, the x, and y resolutions must exist as
    tifffile.tifffile.TiffTags entries as 'XResolution' and 'YResolution'
    respectively. If these values cannot be found, a value of 1 for the
    corresponding dimension is returned.

    Returns
    -------
    res : np.ndarray
        Resolution of the image in ZYX order.
    '''

    try:
        if isinstance(file_.pages.pages[0].tags['XResolution'], tifffile.TiffTag):
            x = file_.pages.pages[0].tags['XResolution'].value[0]
        else:
            x = file_.pages.pages[0].tags['XResolution']
    except (KeyError, TypeError):
        try:
            x = file_.metadata['voxelsizex']
        except (KeyError, TypeError):
            x = 1.0
    try:
        if isinstance(file_.pages.pages[0].tags['YResolution'], tifffile.TiffTag):
            y = file_.pages.pages[0].tags['YResolution'].value[0]
        else:
            y = file_.pages.pages[0].tags['YResolution']
    except (KeyError, TypeError):
        try:
            y = file_.metadata['voxelsizey']
        except (KeyError, TypeError):
            y = 1.0
    try:
        z = file_.metadata['spacing']
    except (KeyError, TypeError):
        try:
            z = file_.metadata['voxelsizez']
        except (KeyError, TypeError):
            z = 1.0

    res = np.array([z, y, x])

    return res

def set_xresolution(file_, value):
    ''' Returns resolution of the image in ZYX format.

    Parameters
    ----------
    file_ : tifffile.TiffFile
        Input file.
    value : float
        New resolution.

    Returns
    -------
    No return. Modifies in place.
    '''
    for ii in xrange(len(file_.pages)):
        if isinstance(file_.pages.pages[ii], tifffile.TiffPage):
            file_.pages.pages[ii].tags['XResolution'] = value


def set_yresolution(file_, value):
    ''' Sets resolution of image file.

    Parameters
    ----------
    file_ : tifffile.TiffFile
        Input file.
    value : float
        New resolution.

    Returns
    -------
    No return. Modifies in place.
    '''
    for ii in xrange(len(file_.pages)):
        if isinstance(file_.pages.pages[ii], tifffile.TiffPage):
            file_.pages.pages[ii].tags['YResolution'] = value


def set_zresolution(file_, value):
    ''' Sets resolution of image file.

    Parameters
    ----------
    file_ : tifffile.TiffFile
        Input file.
    value : float
        New resolution.

    Returns
    -------
    No return. Modifies in place.
    '''
    file_.imagej_metadata['spacing'] = value


def set_resolution(file_, values):
    ''' Sets resolution of image file.

    Parameters
    ----------
    file_ : tifffile.TiffFile
        Input file.
    values : np.ndarray
        New resolution.

    Note
    ----
    Values are in order ZYX.

    Returns
    -------
    No return. Modifies in place.
    '''

    if len(values) != 3:
        # TODO: Should this be IOError?
        raise IOError('Resolution must be of length 3.')

    set_zresolution(file_, values[0])
    set_yresolution(file_, values[1])
    set_xresolution(file_, values[2])


def get_metadata(file_, mtype=''):
    ''' Retrieve file metadata.

    Parameter
    ---------
    file_ : tifffile.TiffFile
        Input file.
    mtype : str, optional
        Metadata type. Possible values are among others 'lsm' and 'imagej'.
        (Default = 'imagej').

    Return
    ------
    metadata : dict
        File metadata.
    '''
    if mtype != '':
        metadata = eval('file_.%s_metadata' % mtype)
    else:
        metadata = file_.metadata
    return metadata


def set_metadata(file_, metadata, mtype='imagej'):
    ''' Set file metadata.

    Parameter
    ---------
    file_ : tifffile.TiffFile
        Input file.
    metadata : dict
        New metadata dictionary.
    mtype : str, optional
        Metadata type. Possible values are among others 'lsm' and 'imagej'.
        (Default = 'imagej').

    Note
    ----
    Missing but required values will be added when the file is saved if not
    contained in ultimate metadata. See @tiffsave.

    Return
    ------
    No return. Modifies input file in place.
    '''
    if mtype != '':
        exec('file_.%s_metadata = metadata' % mtype)
    else:
        file_.metadata = metadata

def tvf2si(file_, reduce_dims=[]):
    ''' Convert a TiffFile to a SpatialImage.

    Parameter
    ---------
    file_ : tifffile.TiffFile
        Input file.
    reduce_dims : list
        List of tuples setting what dimensions to reduce, and by what index,
        using np.take under the hood. First tuple index sets the index of the
        given axis, whereas the second tuple index sets the corresponding axis
        itself. An input array arr of shape (193, 3, 512, 512) using
        reduce_dims=[(2, 1)] therefore inputs the equivalent of arr[:,2,...]
        into the conversion.

    Return
    ------
    simage : timagetk.components.SpatialImage
        SpatialImage object.
    '''

    try:
        from timagetk.components import SpatialImage
    except:
        print('Requires working version of timagetk.')

    metadata = file_.metadata
    arr = file_.data

    if len(reduce_dims) > 0:
        reduce_dims = sorted(reduce_dims, key=lambda l: l[0], reverse=True)
        for ii in xrange(len(reduce_dims)):
            arr = np.take(arr, reduce_dims[ii][0], axis=reduce_dims[ii][1])

    simage = SpatialImage(arr, metadata_dict=metadata,
                          voxelsize=get_resolution(file_))

    if not simage:
        raise Exception('Image not created. This is typically due to the input' +
                        ' image having too many dimensions. Try reducing using' +
                        ' input parameter "reduce_dims".')

    return simage

def si2tvf(file_, **kwargs):
    ''' TODO: This might not be working properly yet. '''
    try:
        from timagetk.components import SpatialImage
        import tempfile
    except:
        print('Requires working version of timagetk. Also requires tempfile.')

    metadata = file_.metadata
    resolution = eval(str(file_.metadata['voxelsize']))

    # These can mess up reading/writing TiffFiles
    to_delete = ['shape', 'dim', 'type', 'voxelsize', 'extent', 'max', 'mean', 'min']
    for ii in to_delete:
        if ii in metadata:
            del metadata[ii]

    # TODO: Make it so that we create a TiffFile directly from the array.
    tmpname = tempfile.NamedTemporaryFile().name
    tiffsave(tmpname, file_.get_array(), metadata=metadata,
             resolution=resolution, **kwargs)
    return tiffload(tmpname)
