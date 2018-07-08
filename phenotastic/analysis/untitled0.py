#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:11:53 2018

@author: henrik
"""
import numpy as np

files = np.array(["Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-2",
                  "Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-4",
                  "Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-5",
                  "Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-10",
                  "Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-1",
                  "Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-3",
                  "Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-9",
                  "Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-7",
                  "Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-6",
                  "Col0-Seeds-LowN-24h-light-1-2-Soil-1-2-Sand-LowN-8",
                  "Col0-Seeds-LowN-24h-light-1-3-Soil-2-3-Sand-LowN-7",
                  "Col0-Seeds-LowN-24h-light-1-3-Soil-2-3-Sand-LowN-1",
                  "Col0-Seeds-LowN-24h-light-1-3-Soil-2-3-Sand-LowN-4",
                  "Col0-Seeds-LowN-24h-light-1-3-Soil-2-3-Sand-LowN-2",
                  "Col0-Seeds-LowN-24h-light-1-3-Soil-2-3-Sand-LowN-5",
                  "Col0-Seeds-LowN-24h-light-1-3-Soil-2-3-Sand-LowN-3",
                  "Col0-Seeds-LowN-24h-light-1-3-Soil-2-3-Sand-LowN-6",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-7",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-6",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-2",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-3",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-10",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-4",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-5",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-8",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-1",
                  "Col0-Seeds-LowN-24h-light-1-Soil-0-Sand-LowN-9",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-1",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-2",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-3",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-4",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-5",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-6",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-7",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-8",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-9",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-10",
                  "Col0-Seeds-LowN-24h-light-2-3--Soil-1-3-Sand-LowN-11"])

scores = np.array([1, 1, 3, 1, 2, 1, 2, 3, 2, 1, 1, 1, 1, 2, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1])
files = files[scores == 1]


import pandas as pd
import os
data = pd.read_csv('/home/henrik/out_fib_comparison/meristem_data.dat', sep="\t")
names = np.array(map(lambda x: os.path.basename(x)[:-4], data.fname.values))

data = data.loc[np.isin(names, files)]
from ggplot import *
#data


ggplot(aes(x='angle_shifted', y='fname', color='fname'), data=data) + \
    geom_point() + \
    theme_bw()
#    facet_wrap('fname') + \


