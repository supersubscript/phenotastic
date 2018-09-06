#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:26:44 2018

@author: henrik
"""

import tifffile as tiff
import javabridge as jv, bioformats as bf
import numpy as np
import phenotastic.file_processing as fp
import os
import untangle
from phenotastic.misc import mkdir

jv.start_vm(class_path=bf.JARS, max_heap_size='6G')

fname = '/home/henrik/data/from-marcus/R2D2 4DAS segmentation and quantification trial_for sending to Henrik_20 august 2018.lif'
OUTPATH = '/home/henrik/data/from_marcus/lif-converted/'
mkdir(OUTPATH)


rdr = bf.ImageReader(path=fname, perform_init=True)
nz = rdr.rdr.getSizeZ()
nx = rdr.rdr.getSizeX()
ny = rdr.rdr.getSizeY()
nc = rdr.rdr.getSizeC()
nt = rdr.rdr.getSizeT()
ns = rdr.rdr.getSeriesCount()

omexml = bf.get_omexml_metadata(fname).encode('utf-8')
a = untangle.parse(omexml)

# Lasers same 6 for all instruments
# Microscope same for all
# Lasers same 6 for all instruments
# Detectors different, but only holds zoom value
# Filters different

### Per image
refr_index = [float(a.OME.Image[ii].ObjectiveSettings['RefractiveIndex']) for ii in xrange(ns)]
axes = [str(a.OME.Image[ii].Pixels['DimensionOrder']) for ii in xrange(ns)]
NA = [float(a.OME.Instrument[ii].Objective['LensNA']) for ii in xrange(ns)]
magnification = [float(a.OME.Instrument[ii].Objective['NominalMagnification']) for ii in xrange(ns)]

z = [int(a.OME.Image[ii].Pixels['SizeZ']) for ii in xrange(ns)]
y = [int(a.OME.Image[ii].Pixels['SizeY']) for ii in xrange(ns)]
x = [int(a.OME.Image[ii].Pixels['SizeX']) for ii in xrange(ns)]
c = [int(a.OME.Image[ii].Pixels['SizeC']) for ii in xrange(ns)]

voxelsizey = [float(a.OME.Image[ii].Pixels['PhysicalSizeY']) for ii in xrange(ns)]
voxelsizex = [float(a.OME.Image[ii].Pixels['PhysicalSizeX']) for ii in xrange(ns)]
voxelsizez = [float(a.OME.Image[ii].Pixels['PhysicalSizeZ']) for ii in xrange(ns)]

### Per image, per channel
# TODO: Must identify which detector for which channel
filter_used = np.array([[int(a.OME.Image[ii].Pixels.Channel[jj].LightPath.EmissionFilterRef['ID'][-1])
                         for jj in xrange(c[ii])] for ii in xrange(ns)])
cutins = np.array([[float(a.OME.Instrument[ii].Filter[jj].TransmittanceRange['CutIn'])
                    for jj in filter_used[ii]] for ii in xrange(ns)])
cutouts = np.array([[float(a.OME.Instrument[ii].Filter[jj].TransmittanceRange['CutOut'])
                     for jj in filter_used[ii]] for ii in xrange(ns)])
em_wavelens = np.array(
    [np.array(zip(cutins, cutouts))[ii].T for ii in xrange(ns)])
ex_wavelens = np.array([[a.OME.Image[ii].Pixels.Channel[jj]['ExcitationWavelength']
                         for jj in xrange(nc)] for ii in xrange(ns)], dtype=np.float)

pinhole_diameter = np.array([[float(a.OME.Image[ii].Pixels.Channel[jj]['PinholeSize'])
                         for jj in xrange(nc)] for ii in xrange(ns)])

### Get actual data

for ss in xrange(ns):
    meta = {'voxelsizex' : voxelsizex[ss],
            'voxelsizey' : voxelsizey[ss],
            'voxelsizez' : voxelsizez[ss],
            'spacing' : voxelsizez[ss],
            'pinhole_diameter' : {jj : pinhole_diameter[ss][jj] for jj in xrange(nc)},
            'ex_wavelens' : {jj : ex_wavelens[ss][jj] for jj in xrange(nc)},
            'em_wavelens_start' : {jj : em_wavelens[ss][jj][0] for jj in xrange(nc)},
            'em_wavelens_end' : {jj : em_wavelens[ss][jj][1] for jj in xrange(nc)}
            }
    for tt in xrange(nt):
        result = np.zeros((nz, nc, ny, nx))
        for zz in xrange(nz):
            print ss, tt, zz
            try:
                plane = rdr.read(z=zz, t=tt, series=ss, rescale=False)
                plane = np.transpose(plane, (2, 1, 0))
            except:
                plane = np.zeros((nc, ny, nx))
            result[zz] = plane
        new_fname = os.path.join(OUTPATH, os.path.splitext(
            ('S%dT%d-') % (ss, tt) + os.path.basename(fname))[0] + '.tif')
        fp.tiffsave(new_fname, data=result, metadata=meta, resolution=[meta['spacing'],
                                                                       meta['voxelsizey'],
                                                                       meta['voxelsizex']])




