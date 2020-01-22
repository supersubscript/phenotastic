#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#######
#
# File author(s): Max BRAMBACH <max.brambach.0065@student.lu.se>
#                 Henrik ÅHL <henrik.aahl@slcu.cam.ac.uk>
# Copyright (c) 2017, Max Brambach, Henrik Åhl
# All rights reserved.
# * Redistribution and use in source and binary forms, with or without
# * modification, are not permitted.
#

import sys, os
os.chdir('/home/henrik/projects/surface_extraction/code/phenotastic/phenotastic')

    ############################################################################
import numpy as np
import os
from pycostanza.misc import mkdir

class surface(object):
    """
    A python class for the automated extraction of SAM features in three
    dimensions from a stack of confocal microscope images.

    AutoPhenotype is a python class for the automated extraction of features of
    a shoot apical meristem (SAM) from a stack of confocal microscope images.
    Its main features are:

    * Extraction of the SAMs surface using active contour without edges (ACWE).
    * Separation of the SAMs features into primordiae and the meristem.
    * Evaluation of the features in terms of location, size and divergence angle.

    Parameters
    ----------

    Attributes
    ----------
    data : 3D intensity array, optional
        Image data e.g. from .tif file. Default = None.

    contour : 3D boolean array, optional
        Contour generated from data. Default = None.

    mesh : vi.PolyData, optional
        Mesh generated from contour. Can contain scalar values. Default = None.

    pdata : pd.DataFrame, optional
        Point data. Default = None.

    ddata : pd.DataFrame, optional
        Domain data. Default = None.

    Returns
    -------


    """

    def __init__(self, data=None, contour=None, mesh=None,
                 pdata=None, ddata=None):
        """Builder"""
        self.data = data
        self.contour = contour
        self.mesh = mesh
        self.pdata = pdata
        self.ddata = ddata

    def save(self, path, sep='\t',otype='ply', mode='binary'):
        """
        Saves the all data stored in an AutoPhenotype Object.

        Parameters
        ----------
        path :


        Returns
        -------


        """
        mkdir(path)

        if self.pdata is not None:
            self.pdata.to_csv(os.path.join(path, 'point_data.csv'), sep=sep, index=False)
        if self.ddata is not None:
            self.ddata.to_csv(os.path.join(path, 'domain_data.csv'), sep=sep, index=False)

        if self.data is not None:
            np.save(os.path.join(path, 'data.npy'), self.data)
        if self.contour is not None:
            np.save(os.path.join(path, 'contour.npy'), self.contour)

        if self.mesh is not None:
            self.mesh.Save(os.path.join(path, 'mesh.ply'), otype=otype, mode=mode)

    def load(self, path):
        raise Exception('"load" method not yet implemented.')
