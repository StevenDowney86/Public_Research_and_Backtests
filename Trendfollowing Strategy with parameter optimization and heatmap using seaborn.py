#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:16:11 2019

@author: downey
"""

import numpy as np
import pandas as pd
import datetime as dt
from pylab import mpl, plt
from pandas_datareader import data, wb
from pandas_datareader.data import DataReader
import pandas_datareader as pdr
import os
from datetime import datetime
import pandas_datareader.data as web
import seaborn as sns
import matplotlib.pyplot as plot
import quandl as quandl

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

start=datetime(1900, 1, 1)
end=datetime(2019, 1, 1)
              

SP500 = web.DataReader("^GSPC", "av-daily-adjusted", start, end, access_key = 'keyhere') 
SP500 = SP500[['close']]


raw.tail()
SP500.columns = ['GSPC.O']
raw = SP500

symbol = ['GSPC.O']

data = (pd.DataFrame(raw[symbol]).dropna())

SMA1 = 42
SMA2 = 252

data['SMA1'] = data[symbol].rolling(SMA1).mean()
data['SMA2'] = data[symbol].rolling(SMA2).mean()


data.plot(figsize = (10, 6))

data.dropna(inplace = True)

data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)

data.tail()

ax = data.plot(secondary_y = 'Position', figsize = (10,6))
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
data['Strategy'] = data['Position'].shift(1) * data['Returns']

data.round(4).head()

data.dropna(inplace = True)

np.exp(data[['Returns', 'Strategy']].sum())

data[['Returns', 'Strategy']].std() * 252 ** 0.5

ax = data[['Returns', 'Strategy']].cumsum(
        ).apply(np.exp).plot(figsize = (10, 6))
data['Position'].plot(ax = ax, secondary_y = 'Position', style = '--')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

from itertools import product

sma1 = range(20, 61, 4)
sma2 = range(180, 281, 10)

results = pd.DataFrame()
for SMA1, SMA2 in product(sma1, sma2):
    data = pd.DataFrame(raw[symbol])
    data.dropna(inplace = True)
    data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
    data['SMA1'] = data[symbol].rolling(SMA1).mean()
    data['SMA2'] = data[symbol].rolling(SMA2).mean()
    data.dropna(inplace = True)
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data.dropna(inplace = True)
    perf = np.exp(data[['Returns', 'Strategy']].sum())
    results = results.append(pd.DataFrame(
                {'SMA1': SMA1, 'SMA2': SMA2,
                 'MARKET': perf['Returns'],
                 'STRATEGY': perf['Strategy'],
                 'OUT': perf['Strategy'] - perf['Returns']},
                 index = [0]), ignore_index = True)

results.info()
results.sort_values('OUT', ascending = False).head(15)
results.head()
heatmapresults = results[["SMA1","SMA2","STRATEGY"]]
heatmapresults.head()
heat_map = sns.heatmap(heatmapresults)
sns.heatmap(results.iloc[:,0:1])


#create heatmap for trend speeds and performance
heatmapresults = heatmapresults.pivot("SMA1","SMA2","STRATEGY")
heatmapresults.head()
ax = sns.heatmap(heatmapresults, annot = True)

outperformance = results[["SMA1","SMA2","OUT"]]
outperformanceresults = outperformance.pivot("SMA1","SMA2","OUT")
ax = sns.heatmap(outperformanceresults, annot = True, linewidths=.5,
                 cmap="YlGnBu")


######################################################################

#looking at other data
Gold = quandl.get("LBMA/GOLD", authtoken="keyhere")
Gold = Gold[['USD (AM)']]
Gold.columns = ['Gold USD']
raw = Gold

symbol = ['Gold USD']

sma1 = range(5, 80, 4)
sma2 = range(100, 300, 10)

results2 = pd.DataFrame()
for SMA1, SMA2 in product(sma1, sma2):
    data = pd.DataFrame(raw[symbol])
    data.dropna(inplace = True)
    data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
    data['SMA1'] = data[symbol].rolling(SMA1).mean()
    data['SMA2'] = data[symbol].rolling(SMA2).mean()
    data.dropna(inplace = True)
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, 0)
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data.dropna(inplace = True)
    perf = np.exp(data[['Returns', 'Strategy']].sum())
    results2 = results2.append(pd.DataFrame(
                {'SMA1': SMA1, 'SMA2': SMA2,
                 'MARKET': perf['Returns'],
                 'STRATEGY': perf['Strategy'],
                 'OUT': perf['Strategy'] - perf['Returns']},
                 index = [0]), ignore_index = True)

results2.info()
results2.sort_values('OUT', ascending = False).head(15)

outperformance_GOLD = results2[["SMA1","SMA2","OUT"]]
outperformanceresults_GOLD = outperformance_GOLD.pivot("SMA1","SMA2","OUT")
ax = sns.heatmap(outperformanceresults_GOLD, annot = False, linewidths=.5,
                 cmap="Reds")


