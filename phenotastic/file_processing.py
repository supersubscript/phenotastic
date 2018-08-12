#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:56:10 2018

@author: henrik
"""

import numpy as np
from tifffile import TiffFile, imsave, tifffile

class tvfile(TiffFile):
    """
    Namespace storing the corresponding file stream and attributes governing
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

    Returns
    -------

    """

    def __init__(self, fname='', data=None, metadata=None):
        if fname != '':
            super(tvfile, self).__init__(fname)
        self.data = data
        self.metadata = metadata

def tiffsave(fname, data, metadata=None, resolution=[1., 1., 1.], dtype=None,
             unit='um', segmented=False, backgroundlabel='', walllabel='', ncells='',
             segmentation_software='', segmentation_method='', parameters='',
             preprocessing='', extra_tags={},  **kwargs):
    """
    Save a file in ImageJ-compatible Tiff format.

    Parameters
    ----------
    fname : str,
        File path.

    data : np.ndarray
        Data to save.

    metadata : dict, optional
        File metadata to store in file (ImageJ format). Default = None.

    resolution : 3-tuple, optional
         Resolution of the file. Default value = [1,1,1]. Ordered as ZYX.

    Returns
    -------
    No return. Saves file to disk.

    """
    metadata = {} if not metadata else metadata.copy()

    # TODO: Figure out what to do with this
    if resolution[0] < 1e-4:
        resolution[0] *= 1e6
        metadata['spacing'] = resolution[0]
    if 'voxelsizez' in metadata:
        metadata['voxelsizez'] = resolution[0]

    if resolution[1] < 1e-4:
        resolution[1] *= 1e6
    if 'voxelsizey' in metadata:
        metadata['voxelsizey'] = resolution[1]

    if resolution[2] < 1e-4:
        resolution[2] *= 1e6
    if 'voxelsizex' in metadata:
        metadata['voxelsizex'] = resolution[2]

    metadata.setdefault('unit', unit)
    metadata.setdefault('slices', data.shape[0])
    metadata['segmented'] = segmented
    metadata['axes'] = 'ZCYX'

    if data.ndim == 4:
        metadata['channels'] = data.shape[1]
        if 'dimensionz' in metadata:
            metadata['dimensionz'] = data.shape[0]
        if 'dimensionchannels' in metadata:
            metadata['dimensionchannels'] = data.shape[1]
        if 'dimensiony' in metadata:
            metadata['dimensiony'] = data.shape[2]
        if 'dimensionx' in metadata:
            metadata['dimensionx'] = data.shape[3]

    elif data.ndim == 3:
        metadata['channels'] = 1
        if 'dimensionz' in metadata:
            metadata['dimensionz'] = data.shape[0]
        if 'dimensionchannels' in metadata:
            metadata['dimensionchannels'] = 1
        if 'dimensiony' in metadata:
            metadata['dimensiony'] = data.shape[1]
        if 'dimensionx' in metadata:
            metadata['dimensionx'] = data.shape[2]
    else:
        raise Exception('Invalid data structure')

    if metadata['segmented']:
        metadata['backgroundlabel'] = segmented
        metadata['walllabel'] = walllabel
        metadata['segmentation_software'] = segmentation_software
        metadata['segmentation_method'] = segmentation_method
        metadata['parameters'] = parameters
        metadata['preprocessing'] = preprocessing

    for ii in extra_tags:
        metadata[ii] = extra_tags[ii]

    imsave(fname, data=data, shape=data.shape, dtype=data.dtype,
           imagej=True, resolution=[1./resolution[2], 1./resolution[1]],
           metadata=metadata)


    #           **kwargs)
#            code : int
#                The TIFF tag Id.
#            dtype : str
#                Data type of items in 'value' in Python struct format.
#                One of B, s, H, I, 2I, b, h, i, 2i, f, d, Q, or q.
#            count : int
#                Number of data values. Not used for string or byte string
#                values.
#            value : sequence
#                'Count' values compatible with 'dtype'.
#                Byte strings must contain count values of dtype packed as
#                binary data.
#            writeonce : bool
#                If True, the tag is written to the first page only.
#    282: 'XResolution',
#    283: 'YResolution',

def tiffload(fname):
    """
    Loads an image file as a tifffile.TiffFile object.

    Parameters
    ----------
    fname : str
        File path.

    Returns
    -------
    fobj : tvfile
        The file object.

    """
    fobj = tvfile(fname) #TiffFile(fname)

    def lowerdict(d):
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
    # TODO: Make sure this is OK for all cases! Unsure how channels/series etc are treated within the array of pages
    if nseries > 1:
        newSeries = tifffile.TiffPageSeries(
            pages[::nseries], (npages / nseries, nchannels, pages[0].imagelength,
                               pages[0].imagewidth), dtype=ss.dtype, axes='ZCYX',
        stype='ImageJ')
    else:
        newSeries = tifffile.TiffPageSeries(
            pages[::nseries], (npages / nchannels, nchannels, pages[0].imagelength,
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
    """
    TODO

    Parameters
    ----------
    fobj :

    channel :
         (Default value = -1)

    Returns
    -------

    """

    arr = fobj.asarray()
    arr = tifffile.tifffile.transpose_axes(arr, fobj.series[0].axes, 'TZCYXS')
    if channel > -1:
        arr = arr[0, :, channel, : ,:, 0]
    else:
        arr = arr[0, :, :, : ,:, 0]
    return arr

def get_resolution(fobj):
    """
    Returns resolution of the image in ZYX order.

    The z-resolution must have been stored as imagej metadata under the name
    "spacing". Similarly, the x, and y resolutions must exist as
    tifffile.tifffile.TiffTags entries as "'"XResolution"'" and "YResolution"
    respectively. If these values cannot be found, a value of 1 for the
    corresponding dimension is returned.

    Parameters
    ----------
    fobj : tifffile.TiffFile

    Returns
    -------


    """

    try:
        if isinstance(fobj.pages.pages[0].tags['XResolution'], tifffile.TiffTag):
            x = fobj.pages.pages[0].tags['XResolution'].value[0]
        else:
            x = fobj.pages.pages[0].tags['XResolution']
    except (KeyError, TypeError):
        try:
            x = fobj.metadata['voxelsizex']
        except (KeyError, TypeError):
            x = 1.0
    try:
        if isinstance(fobj.pages.pages[0].tags['YResolution'], tifffile.TiffTag):
            y = fobj.pages.pages[0].tags['YResolution'].value[0]
        else:
            y = fobj.pages.pages[0].tags['YResolution']
    except (KeyError, TypeError):
        try:
            y = fobj.metadata['voxelsizey']
        except (KeyError, TypeError):
            y = 1.0
    try:
        z = fobj.metadata['spacing']
    except (KeyError, TypeError):
        try:
            z = fobj.metadata['voxelsizez']
        except (KeyError, TypeError):
            z = 1.0

    res = np.array([z, y, x])

    return res

def set_xresolution(fobj, value):
    """
    Returns resolution of the image in ZYX format.

    Parameters
    ----------
    fobj : tifffile.TiffFile
        Input file.

    value : float
        New resolution.

    Returns
    -------


    """
    for ii in xrange(len(fobj.pages)):
        if isinstance(fobj.pages.pages[ii], tifffile.TiffPage):
            fobj.pages.pages[ii].tags['XResolution'] = value


def set_yresolution(fobj, value):
    """Sets resolution of image file.

    Parameters
    ----------
    fobj : tifffile.TiffFile
        Input file.

    value : float
        New resolution.

    Returns
    -------


    """
    for ii in xrange(len(fobj.pages)):
        if isinstance(fobj.pages.pages[ii], tifffile.TiffPage):
            fobj.pages.pages[ii].tags['YResolution'] = value


def set_zresolution(fobj, value):
    """
    Sets resolution of image file.

    Parameters
    ----------
    fobj : tifffile.TiffFile
        Input file.

    value : float
        New resolution.

    Returns
    -------


    """
    fobj.imagej_metadata['spacing'] = value


def set_resolution(fobj, values):
    """
    Sets resolution of image file.

    Parameters
    ----------
    fobj : tifffile.TiffFile
        Input file.

    values : np.ndarray
        New resolution.

    Note
    ----
    Values are in order ZYX.

    Returns
    -------

    """
    if len(values) != 3:
        # TODO: Should this be IOError?
        raise IOError('Resolution must be of length 3.')

    set_zresolution(fobj, values[0])
    set_yresolution(fobj, values[1])
    set_xresolution(fobj, values[2])


def get_metadata(fobj, mtype=''):
    """
    Retrieve file metadata.

    Parameters
    ----------
    fobj : tvfile
        Input file.

    mtype : str, optional
        Metadata type. Possible values are among others "lsm2 and ""imagej".
        (Default = "imagej").

    Returns
    -------
    metadata : dict
        File metadata.

    """
    if mtype != '':
        metadata = eval('fobj.%s_metadata' % mtype)
    else:
        metadata = fobj.metadata
    return metadata


def set_metadata(fobj, metadata, mtype='imagej'):
    """
    Set file metadata.

    Parameters
    ----------
    fobj : tifffile.TiffFile
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

    Returns
    -------
        No return. Modifies input file in place.

    """
    if mtype != '':
        exec('fobj.%s_metadata = metadata' % mtype)
    else:
        fobj.metadata = metadata

def tvf2si(fobj, reduce_dims=[]):
    """
    Convert a TiffFile to a SpatialImage.

    Parameters
    ----------
    fobj : tifffile.TiffFile
        Input file.

    reduce_dims : list
        List of tuples setting what dimensions to reduce, and by what index,
        using np.take under the hood. First tuple index sets the index of the
        given axis, whereas the second tuple index sets the corresponding axis
        itself. An input array arr of shape (193, 3, 512, 512) using
        reduce_dims=[(2, 1)] therefore inputs the equivalent of arr[:,2,...]
        into the conversion.

    Returns
    -------
        simage : timagetk.components.SpatialImage
            SpatialImage object.

    """

    try:
        from timagetk.components import SpatialImage
    except:
        print('Requires working version of timagetk.')

    metadata = fobj.metadata
    arr = fobj.data

    if len(reduce_dims) > 0:
        reduce_dims = sorted(reduce_dims, key=lambda l: l[0], reverse=True)
        for ii in xrange(len(reduce_dims)):
            arr = np.take(arr, reduce_dims[ii][0], axis=reduce_dims[ii][1])

    simage = SpatialImage(arr, metadata_dict=metadata,
                          voxelsize=get_resolution(fobj))

    if not simage:
        raise Exception('Image not created. This is typically due to the input' +
                        ' image having too many dimensions. Try reducing using' +
                        ' input parameter "reduce_dims".')

    return simage

def si2tvf(fobj, **kwargs):
    """
    TODO: This might not be working properly yet.

    Parameters
    ----------
    fobj :

    Returns
    -------

    """
    try:
        from timagetk.components import SpatialImage
        import tempfile
    except:
        print('Requires working version of timagetk. Also requires tempfile.')

    metadata = fobj.metadata
    resolution = eval(str(fobj.metadata['voxelsize']))

    # These can mess up reading/writing TiffFiles
    to_delete = ['shape', 'dim', 'type', 'voxelsize', 'extent', 'max', 'mean', 'min']
    for ii in to_delete:
        if ii in metadata:
            del metadata[ii]

    # TODO: Make it so that we create a TiffFile directly from the array.
    tmpname = tempfile.NamedTemporaryFile().name
    tiffsave(tmpname, fobj.get_array(), metadata=metadata,
             resolution=resolution, **kwargs)
    return tiffload(tmpname)
