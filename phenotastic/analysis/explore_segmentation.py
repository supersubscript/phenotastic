#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:46:59 2018

@author: henrik
"""
import pandas as pd
import numpy as np
import os
from ggplot import *
import seaborn as sb

sdata = pd.read_csv('/home/henrik/out_fib_comparison_corrected/ratings.dat', sep='\t',
                   header=None, names=['score', 'name'])
sdata= sdata[sdata.score == 1]
sdata['name'] = sdata.name.map(lambda x: x[:-8] + '.lsm')

mdata = pd.read_csv('/home/henrik/out_fib_comparison_corrected/meristem_data.dat', sep='\t')
mdata['fname'] = mdata.fname.map(lambda x: os.path.basename(x))

data = mdata.iloc[np.isin(mdata.fname, sdata.name)]

#melted = pd.melt(data, id_vars='fname', value_vars=('dist_boundary', 'dist_com', 'area', 'maxdist', 'maxdist_xy'))

''' Areas '''
#sb.pairplot(data.loc[np.logical_not(data.ismeristem)][['fname', 'dist_boundary',
#                     'dist_com', 'area', 'maxdist', 'maxdist_xy']],
#hue='fname', diag_kws={'bins': 25}, plot_kws={'alpha': 0.7})

''' Angles '''
data = data.sort_values(by=['fname', 'domain'])
#sb.stripplot(y='fname', x = 'angle', data=data, hue='domain')

names = sdata.name.values

GOLDEN_ANGLE = 137.5
FULL_PERIOD = 360.0
orders = []
for ii in names:
    d = data.loc[np.isin(data.fname, ii)]
    not_meristem = d[np.logical_not(d.ismeristem)]
    angles = np.array([FULL_PERIOD - jj if np.abs(FULL_PERIOD - jj - GOLDEN_ANGLE) < np.abs(
        jj - GOLDEN_ANGLE) else jj for jj in np.abs(np.diff(not_meristem.angle.values))])

    bestorder = np.append(0, not_meristem.domain.values)
    for jj in xrange(1, len(not_meristem) + 1):
        order = [0, jj]
        last_angle = not_meristem.loc[not_meristem.domain == jj].angle.values

        # One direction
        for kk in xrange(len(not_meristem) - 1):
            remainder = not_meristem.loc[np.logical_not(np.isin(not_meristem.domain, order))]
            angles_left = remainder.angle.values
            next_ = np.argmin(np.min(np.array(
                    zip(np.abs(angles_left - (last_angle + GOLDEN_ANGLE)),
                        np.abs(last_angle + GOLDEN_ANGLE -
                               (angles_left + FULL_PERIOD)))), axis=1))

            last_angle = not_meristem.loc[np.logical_not(np.isin(not_meristem.domain, order))].iloc[next_].angle
            next_ = remainder.iloc[next_].domain
            order = np.append(order, next_)
        res = np.array([360 - ang if np.abs(360 - ang - 137.5) < np.abs(ang - 137.5)
            else ang for ang in np.abs(np.diff(d.iloc[order].angle[1:].values))])


        if np.abs(np.mean(res) - 137.5) < np.abs(np.mean(np.array([360 - ang if
                 np.abs(360 - ang - 137.5) < np.abs(ang - 137.5) else
                 ang for ang in
                 np.abs(np.diff(d.iloc[bestorder].angle[1:].values))])) - 137.5):
            if np.polyfit(xrange(len(d) - 1), d.iloc[order][1:].area.values, 1)[0] < -10000 and np.all(np.isin([1, 2], order[0:4])):
                bestorder = order.copy()

        # Other direction
        order = [0, jj]
        last_angle = not_meristem.loc[not_meristem.domain == jj].angle.values
        for kk in xrange(len(not_meristem) - 1):
            remainder = not_meristem.loc[np.logical_not(np.isin(not_meristem.domain, order))]
            angles_left = remainder.angle.values
            next_ = np.argmin(np.min(np.array(
                    zip(np.abs(angles_left - (last_angle - GOLDEN_ANGLE)),
                        np.abs(last_angle - GOLDEN_ANGLE -
                               (angles_left + FULL_PERIOD)))), axis=1))

            last_angle = not_meristem.loc[np.logical_not(np.isin(not_meristem.domain, order))].iloc[next_].angle
            next_ = remainder.iloc[next_].domain
            order = np.append(order, next_)
        res = np.array([360 - ang if np.abs(360 - ang - 137.5) < np.abs(ang - 137.5)
            else ang for ang in np.abs(np.diff(d.iloc[order].angle[1:].values))])

        if np.abs(np.mean(res) - 137.5) < np.abs(np.mean(
                np.array([360 - ang if np.abs(360 - ang - 137.5) < np.abs(ang - 137.5)
                else ang for ang in np.abs(np.diff(d.iloc[bestorder].angle[1:].values))])) - 137.5):
            if np.polyfit(xrange(len(d) - 1), d.iloc[order][1:].area.values, 1)[0] < -10000 and np.all(np.isin([1, 2], order[0:4])):
                bestorder = order.copy()

    res = np.array([360 - ang if np.abs(360 - ang - 137.5) < np.abs(ang - 137.5)
                else ang for ang in np.abs(np.diff(d.iloc[bestorder].angle[1:].values))])
    orders.append(bestorder)

#data['div_ang'] = np.full(len(data), np.nan)
new_df = pd.DataFrame(columns=data.columns)
for ii, name in enumerate(names):
    new_order = data.loc[data.fname == name].iloc[orders[ii]].copy()
    new_order.div_ang = np.append(np.array([360 - ang if np.abs(360 - ang - 137.5) < np.abs(ang - 137.5)
                else ang for ang in np.abs(np.diff(new_order.angle.values))]), np.nan)
    new_df = new_df.append(new_order, ignore_index=True)

new_df.div_ang.hist(bins=80)
#sb.palplot()
cmap = sb.cubehelix_palette(light=1, as_cmap=False)
sb.set()
#sb.set_palette('deep')
sb.set_style("white")
sb.set(font_scale = 2)
g = sb.jointplot(x="area", y="div_ang", data=new_df, kind="kde", cut=0, xlim=[0, new_df.area.max()], space=0.1, size=8, ratio=5, marginal_kws=dict(color='purple')).set_axis_labels('Area (arbitrary units)', 'Divergence angle (degrees)')
#for ax in g.axes.flat

#dif
#g = sb.jointplot(x="area", y="div_ang", data=new_df, kind="kde", color="m")
#g.plot_joint(sb.scatter, c="w", s=30, linewidth=1, marker="+")




