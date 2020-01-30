#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:49:16 2020

@author: downey
"""
'''This came from a curiosity to see if Fed QE caused, was correlated with SPY
returns, and could be used as a market timing strategy'''

import numpy as np
import pandas as pd
from pylab import mpl, plt
import seaborn as sns
from datetime import datetime
import pandas_datareader.data as web
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns',110)
pd.set_option('display.max_rows',1000)
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

start=datetime(2008, 1, 1)
end=datetime(2019, 12, 31)

Security = web.DataReader("SPY", "av-daily-adjusted", start, end, access_key = 'apikeyhere')                                   

Fed_Balance_Sheet = web.get_data_fred('WALCL', start, end)
Fed_Balance_Sheet.columns = ['Fed Balance Sheet']
Fed_Balance_Sheet['Fed Balance Sheet Change'] = Fed_Balance_Sheet.pct_change(1)

data = Fed_Balance_Sheet.join(Security)
data2 = data[['Fed Balance Sheet', 'Fed Balance Sheet Change','adjusted close']]
data2['SPY returns'] = data2['adjusted close'].pct_change()
  
fed_change = range(1, 30, 1)
spy_change = range(1, 30, 1)
future_spy = range(-30, -1, 1)
corr_results = pd.DataFrame()
for Fed_Change, SPY_change, Future_SPY in product(fed_change,spy_change, future_spy):
    data_new = data2[['Fed Balance Sheet','adjusted close']]
    data_new['Fed BS Change'] = data_new['Fed Balance Sheet'].pct_change(Fed_Change)
    data_new['SPY Change'] = data_new['adjusted close'].pct_change(SPY_change)
    data_new['SPY Shift'] = data_new['SPY Change'].shift(Future_SPY)
    data_new.dropna(inplace = True)
    correlation_stat = stats.spearmanr(data_new['Fed BS Change'],data_new['SPY Shift'])
    correlation = correlation_stat[0]
    p_value = correlation_stat[1]
    corr_results = corr_results.append(pd.DataFrame(
                {'Fed BS Change': Fed_Change, 'SPY Change': SPY_change, 'SPY Future Shift': Future_SPY,
                 'Corr': correlation, 'p-value': p_value},
                 index = [0]), ignore_index = True)
   
data2.dropna(inplace = True)
stats.spearmanr(data2['Fed Balance Sheet Change'],data2['SPY returns'])
data2.head()

corr_results.sort_values('Corr', ascending = True).tail(15)
corr_results['Corr'].hist()

corr_results.head()
sns.relplot(x="Corr", y="p-value", 
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=corr_results)

corr_results.plot.scatter(x='SPY Future Shift',y='SPY Change',c='Corr', colormap='viridis')
#%%
data3 = data2.dropna()
granger = sm.tsa.stattools.grangercausalitytests(data3[['adjusted close','Fed Balance Sheet']], maxlag = 52, verbose = True)

#####trying to take out p values and lags and put in dataframe and use chart###

lags = range(1, 53,1)
granger_values = pd.DataFrame()

for i in lags:
    df = granger.get(i)
    dicter = df[0]
    pvalue = dicter['ssr_ftest'][1]
    lag = dicter['ssr_ftest'][3]
    granger_values = granger_values.append(pd.DataFrame(
                {'p-value': pvalue, 'Lags': lag},
                 index = [0]), ignore_index = True)

granger_values

ax = granger_values.plot.bar(x = 'Lags', y = 'p-value')
ax.axhline(y=.05, xmin=-1, xmax=1, color='r', linestyle='--', lw=2)
plt.title('Granger Casuality P-Values using F-Test')

#%%
lags = range(1, 53, 1)
threshold = np.linspace(-.3,.3,20)

results = pd.DataFrame()
for Lags, Threshold in product(lags, threshold):
    data = data2[['Fed Balance Sheet','adjusted close']]
    data.dropna(inplace = True)
    data['Fed Balance Sheet Change'] = data['Fed Balance Sheet'].pct_change(Lags)
    data['SPY Returns'] = data['adjusted close'].pct_change()
    data.dropna(inplace = True)
    data['Position'] = np.where(data['Fed Balance Sheet Change'] > Threshold, 1, 0)
    data['Strategy'] = data['Position'].shift(1) * data['SPY Returns']
    data.dropna(inplace = True)
    D = len(data)
    perf = data[['SPY Returns', 'Strategy']].add(1).prod() ** (52 / D) - 1 
    std = data[['SPY Returns', 'Strategy']].std() * 52 ** 0.5
    Sharpe = perf/std
    results = results.append(pd.DataFrame(
                {'MARKET': Sharpe['SPY Returns'],
                 'STRATEGY': Sharpe['Strategy'], 'Alpha': perf[1]-perf[0],
                 'Lags':Lags,'Threshold': np.round(Threshold,2),
                 'OUT': Sharpe['Strategy'] - Sharpe['SPY Returns']},
                 index = [0]), ignore_index = True)


results.sort_values('Alpha', ascending = True).head(15)
results[results['Alpha']>0].shape
results.shape
percent_strategies_alpha = results[results['Alpha']>0].shape[0]/results.shape[0]
percent_strategies_alpha
percent_strategies_higher_Sharpe = results[results['OUT']>0].shape[0]/results.shape[0]
percent_strategies_higher_Sharpe


heatmapresults = results[["Lags","Threshold","Alpha"]]
heatmapresults.head()

#create heatmap for trend speeds and performance
heatmapresults = heatmapresults.pivot("Lags","Threshold","Alpha")
heatmapresults.head()
ax = sns.heatmap(heatmapresults, annot = False, cmap="YlGnBu", vmin = -.15, vmax = .10).set_title('Strategy Alpha')




