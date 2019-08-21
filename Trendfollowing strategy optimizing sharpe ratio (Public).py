#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:16:11 2019

@author: downey
"""

import numpy as np
import pandas as pd
from pylab import mpl, plt
import seaborn as sns

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

raw = pd.read_csv('/Users/downey/Coding/Python/Python for Finance/FF_daily.csv', index_col = 0, parse_dates = True)

#dates from July 1 1926 - October 26 1961
raw = raw.iloc[0:10000,]

raw.tail()
raw.head()
raw.columns = ['Market.Close']
raw.info()

symbol = 'Market.Close'

data = (
    pd.DataFrame(raw[symbol])
    .dropna()
  )


from itertools import product

sma1 = range(1, 61, 3)
sma2 = range(180, 381, 10)

results = pd.DataFrame()
for SMA1, SMA2 in product(sma1, sma2):
    data = pd.DataFrame(raw[symbol])
    data.dropna(inplace = True)
    data['Returns'] = data[symbol].pct_change()
    data['SMA1'] = data[symbol].rolling(SMA1).mean()
    data['SMA2'] = data[symbol].rolling(SMA2).mean()
    data.dropna(inplace = True)
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, 0)
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data.dropna(inplace = True)
    D = len(data)
    perf = data[['Returns', 'Strategy']].add(1).prod() ** (252 / D) - 1 
    std = data[['Returns', 'Strategy']].std() * 252 ** 0.5
    Sharpe = perf/std
    results = results.append(pd.DataFrame(
                {'SMA1': SMA1, 'SMA2': SMA2,
                 'MARKET': Sharpe['Returns'],
                 'STRATEGY': Sharpe['Strategy'],
                 'OUT': Sharpe['Strategy'] - Sharpe['Returns']},
                 index = [0]), ignore_index = True)

results.info()
results.sort_values('OUT', ascending = False).head(15)

#create heatmap for trend speeds and Sharpe Ratio
heatmapresults = results
heatmapresults = heatmapresults.pivot("SMA1","SMA2","STRATEGY")
heatmapresults.head()

ax = sns.heatmap(heatmapresults, annot = False)
plt.title('Sharpe Ratio for Trend Speeds July 1 1926 - October 26 1961')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()


