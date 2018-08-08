#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:57:42 2018

@author: henrik
"""
import pandas as pd
import numpy as np
import os

on = '/home/henrik/out_l1_quant3/data.dat'
off = '/home/henrik/out_l1_quant4/data.dat'

don = pd.read_csv(on, sep='\t', header=None, names=['name', 'vals'])
doff = pd.read_csv(off, sep='\t', header=None, names=['name', 'vals'])

don = don[don['name'].str.contains("^((?!NPA-7).)*$", regex=True)]
don.index = range(len(don))

onplants = np.unique(['-'.join(os.path.basename(don.name[ii]).split('-')[4:6]) for ii in xrange(len(don.name))])
onplantmaxes = []
for ii in onplants:
    filt = np.array(map(lambda x: ii in x, don.name.values))
    plantmax = np.hstack(don.vals[filt]).max()
#    plantmax = don.vals[filt].values[0]
    onplantmaxes.append(plantmax)
    print plantmax
    don.loc[filt, 'vals'] /= plantmax

offplants = np.unique(['-'.join(os.path.basename(doff.name[ii]).split('-')[4:6]) for ii in xrange(len(doff.name))])
offplantmaxes = []
for ii in offplants:
    filt = np.array(map(lambda x: ii in x, doff.name.values))
    plantmax = np.hstack(doff.vals[filt]).max()
#    plantmax = doff.vals[filt].values[0]
    offplantmaxes.append(plantmax)
    print plantmax
    doff.loc[filt, 'vals'] /= plantmax

ot1 = don[::3]
ot2 = don[1::3]
ot3 = don[2::3]
odata_ = np.array([ot1.mean(), ot2.mean(), ot3.mean()])

oft1 = doff[::3]
oft2 = doff[1::3]
oft3 = doff[2::3]

#import matplotlib as plt
#for ii in xrange(len(oft1)):
#    plot(np.array([ot1.vals.values[ii], ot2.vals.values[ii], ot3.vals.values[ii]]), color = 'b')
#    plot(np.array([oft1.vals.values[ii], oft2.vals.values[ii], oft3.vals.values[ii]]), color= 'r')


odata_ = np.array([ot1.mean(), ot2.mean(), ot3.mean()])
ofdata_ = np.array([oft1.mean(), oft2.mean(), oft3.mean()])
#
sdodata_ = np.array([ot1.std(), ot2.std(), ot3.std()])
sdofdata_ = np.array([oft1.std(), oft2.std(), oft3.std()])

import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
ax = axs
#ax = axs[0]
#ax.errorbar(range(3), odata_, yerr=sdodata_, fmt='o')

# With 4 subplots, reduce the number of axis ticks to avoid crowding.
ax.locator_params(nbins=1)

#ax = axs[0]
ax.errorbar(range(3), odata_, yerr=sdodata_, fmt='o', color='b')
ax.set_title('On')
ax.errorbar(np.array(range(3)) + 0.1, ofdata_, yerr=sdofdata_, fmt='o', color='r')
ax.set_title('Off')



#
#plot(odata_)
#plot(ofdata_)
##
#plot(sdodata_)
#plot(sdofdata_)