#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:16:11 2019

@author: downey
"""

"""
I was curious if one could estimate if a trend is overextend if it has some
sort of hyperbolic move in the upward direction, for a long/flat portfolio.
So I wanted to compare the rate of change of a certain length (i.e. 10 days)
vs. the similar length rate of change but shifted back a certain number of days
with the idea being if there is a hyperbolic move then the rate of change at 
present will be greater than the rate of change of the comparison

The idea is to have a SMA trendfollowing strategy as default position but get
out and into cash if there is a hyperbolic move. I also ran a historical backtest
to see if you add a banding window around the value at which you would go to cash

I found that a simple SMA trendfollowing was better on an absolute return
basis. 

"""

import numpy as np
import pandas as pd
from pylab import mpl, plt
import seaborn as sns

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
pd.set_option('display.max_columns',110)
              
raw = pd.read_csv('/Users/downey/Coding/Python/Python for Finance/F-F_Research_Data_Factors_daily.csv', index_col = 0, parse_dates = True)
raw.head()
#drop two columns SMB = Small minus Big market cap, and HML = Value factor (low
#price to book minus high price to book) 
raw = raw.drop(columns=['SMB', 'HML'])
#combine the Market Risk Premium and Risk Free rate to get market return
raw['MKT+RF'] = raw['Mkt-RF']+raw['RF']
#divide values by 100 to get decimal percentage 
raw = raw/100

raw['GSPC.O'] = (1+raw['MKT+RF']).cumprod()
raw.tail()
    
symbol = 'GSPC.O'

from itertools import product

#roc_value = 20
#roc_value_comp = 10


roc_value_comp = range(4, 100, 4)
roc_value_comp2 = range(4, 100, 4)

results = pd.DataFrame()
for VALUE1, VALUE2 in product(roc_value_comp, roc_value_comp2):
    data = pd.DataFrame(raw[symbol])
    data.dropna(inplace = True)
    data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
    data['SMA'] = data[symbol].rolling(200).mean()
    data['Rolling ROC'] = data[symbol].pct_change(VALUE1)
    data.dropna(inplace = True)
    data['SMA Position'] = np.where(data['SMA'] < data[symbol], 1, -1)
    #testing to see if there is an overextension and exponential growth in trend
    data['Rolling ROC position'] = np.where(data['Rolling ROC'] < data['Rolling ROC'].shift(VALUE2), 1, -1)
    #only be invested in the market when uptrend and no trend over extension
    data['Strategy Position'] = np.where(data['SMA Position'] == data['Rolling ROC position'], 1, 0)
    data['Strategy'] = data['Strategy Position'].shift(1) * data['Returns']
    data['SMA Strategy'] = data['SMA Position'].shift(1)* data['Returns']
    data.dropna(inplace = True)
    perf = np.exp(data[['Strategy','SMA Strategy']].sum())
    perf.head()    
    perf = perf**(252/len(data)) - 1
    results = results.append(pd.DataFrame(
                {'ROC Window': VALUE1, 'ROC Comparison': VALUE2,
                 'SMA Stategy': perf['SMA Strategy'],
                 'STRATEGY': perf['Strategy'],
                 'OUT': perf['Strategy'] - perf['SMA Strategy']},
                 index = [0]), ignore_index = True)

results.info()
results.sort_values('OUT', ascending = False).head(15)


#create heatmap for trend speeds and performance
outperformance = results[["ROC Window","ROC Comparison","OUT"]]
outperformanceresults = outperformance.pivot("ROC Window","ROC Comparison","OUT")
ax = sns.heatmap(outperformanceresults, annot = False, linewidths=.5,
                 cmap="YlGnBu")
ax.set(xlabel='Rate of Change Comparison Days vs. Other ROC', ylabel='Rate of Change Window')
ax.set_title('Rate of Change vs. SMA 200 Relative Performance')


###############################With Banding##############################
roc_value_comp = range(4, 100, 6)
roc_value_comp2 = range(4, 100, 6)
roc_value_comp3 = range(1, 4, 1)

results = pd.DataFrame()
for VALUE1, VALUE2, VALUE3 in product(roc_value_comp, roc_value_comp2, roc_value_comp3):
    data = pd.DataFrame(raw[symbol])
    data.dropna(inplace = True)
    data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
    data['SMA'] = data[symbol].rolling(200).mean()
    data['Rolling ROC'] = data[symbol].pct_change(VALUE1)
    data.dropna(inplace = True)
    data['SMA Position'] = np.where(data['SMA'] < data[symbol], 1, -1)
    #testing to see if there is an overextension and exponential growth in trend
    data['Rolling ROC position'] = np.where(data['Rolling ROC'] > (data['Rolling ROC'].shift(VALUE2)*VALUE3), -1, 1)
    #only be invested in the market when uptrend and no trend over extension
    data['Strategy Position'] = np.where(data['SMA Position'] == data['Rolling ROC position'], 1, 0)
    data['Strategy'] = data['Strategy Position'].shift(1) * data['Returns']
    data['SMA Strategy'] = data['SMA Position'].shift(1)* data['Returns']
    data.dropna(inplace = True)
    perf = np.exp(data[['Strategy','SMA Strategy']].sum())
    perf.head()    
    perf = perf**(252/len(data)) - 1
    results = results.append(pd.DataFrame(
                {'ROC Window': VALUE1, 'ROC Comparison': VALUE2,
                 'Banding': VALUE3,
                 'SMA Stategy': perf['SMA Strategy'],
                 'STRATEGY': perf['Strategy'],
                 'OUT': perf['Strategy'] - perf['SMA Strategy']},
                 index = [0]), ignore_index = True)

results.info()
results.sort_values('OUT', ascending = False).head(15)


#create heatmap for trend speeds and performance
outperformance = results[["ROC Window","ROC Comparison","OUT"]]
outperformanceresults = outperformance.pivot("ROC Window","ROC Comparison","OUT")
ax = sns.heatmap(outperformanceresults, annot = False, linewidths=.5,
                 cmap="YlGnBu")
ax.set(xlabel='Rate of Change Comparison Days vs. Other ROC', ylabel='Rate of Change Window')
ax.set_title('Rate of Change vs. SMA 200 Relative Performance')





