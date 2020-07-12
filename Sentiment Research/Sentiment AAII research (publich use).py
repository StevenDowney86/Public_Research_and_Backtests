#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:08:09 2020

@author: downey
"""

import pandas as pd
import quandl
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web
import datetime as dt

### Curious if sentiment data can be use to adjust market position ###

# Maybe using AAII sentiment and scale up or down market exposure?
# Maybe using ML with price and econ data to classify if we are in a recession
# or expansionary time. Or use Sahm Rule?
# 
#use the performance_analysis python file to import functions

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)
plt.style.use('ggplot')
quandl.ApiConfig.api_key = "key here"
#turn off pandas warning for index/slicing as copy warning
pd.options.mode.chained_assignment = None  # default='warn'
#%%
AAII_data = quandl.get("AAII/AAII_SENTIMENT",\
                  start_date="1970-01-01", end_date="2020-07-01")

#%%

#Kenneth French data 
    
len(get_available_datasets())

ds = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1927-01-01')

print(ds['DESCR'])

ds[0].head()
#%%

market_data = ds[0]

market_data['Market'] = market_data['Mkt-RF']+market_data['RF']
market_data = market_data/100
market_data['Market Index'] = (1+market_data['Market']).cumprod()
merge=pd.merge(AAII_data,market_data['Market Index'], how='inner', left_index=True, right_index=True)
merge['Market Return'] = merge['Market Index'].pct_change()


df = pd.DataFrame()
df['min'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(0, interpolation='lower')
df['first'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.1, interpolation='lower')
df['second'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.2, interpolation='lower')
df['third'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.3, interpolation='lower')
df['fourth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.4, interpolation='lower')
df['fifth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.5, interpolation='lower')
df['sixth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.6, interpolation='lower')
df['seventh'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.7, interpolation='lower')
df['eigth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.8, interpolation='lower')
df['ninth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.9, interpolation='lower')
df['max'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(1, interpolation='lower')
df['Bull-Bear Spread'] = AAII_data['Bull-Bear Spread']

new_df=pd.merge(df,merge['Market Return'], how='inner', left_index=True, right_index=True)

start = new_df.index.searchsorted(dt.datetime(1987, 6, 26))
end = new_df.index.searchsorted(dt.datetime(2003, 6, 26))

IN_SAMPLE = new_df.iloc[start:end]

IN_SAMPLE['position'] = np.where((IN_SAMPLE['Bull-Bear Spread'] < IN_SAMPLE['ninth']), 1, \
                        np.where((IN_SAMPLE['Bull-Bear Spread'] > IN_SAMPLE['ninth']), 0,1))

IN_SAMPLE['Portfolio Return'] = IN_SAMPLE['position'] * IN_SAMPLE['Market Return'].shift(1)

IN_SAMPLE['Portfolio Index'] = (1+IN_SAMPLE['Portfolio Return']).cumprod()

IN_SAMPLE['Market Index'] = (1+IN_SAMPLE['Market Return']).cumprod()

IN_SAMPLE[['Market Index','Portfolio Index']].plot()
#%%
market_data = ds[0]

market_data['Market'] = market_data['Mkt-RF']+market_data['RF']
market_data = market_data/100
market_data['Market Index'] = (1+market_data['Market']).cumprod()
market_data.head()
merge=pd.merge(AAII_data,market_data['Market Index'], how='inner', left_index=True, right_index=True)
merge['Market Return'] = merge['Market Index'].pct_change()

df = pd.DataFrame()
df['min'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(0, interpolation='lower')
df['first'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.1, interpolation='lower')
df['second'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.2, interpolation='lower')
df['third'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.3, interpolation='lower')
df['fourth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.4, interpolation='lower')
df['fifth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.5, interpolation='lower')
df['sixth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.6, interpolation='lower')
df['seventh'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.7, interpolation='lower')
df['eigth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.8, interpolation='lower')
df['ninth'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.9, interpolation='lower')
df['max'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(1, interpolation='lower')
df['Bull-Bear Spread'] = AAII_data['Bull-Bear Spread']
df['quantile'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(.9)

new_df=pd.merge(df,merge['Market Return'], how='inner', left_index=True, right_index=True)

start = new_df.index.searchsorted(dt.datetime(2009, 1, 1))
end = new_df.index.searchsorted(dt.datetime(2020, 5, 30))

IN_SAMPLE = new_df.iloc[start:end]

IN_SAMPLE['position'] = np.where((IN_SAMPLE['Bull-Bear Spread'] < IN_SAMPLE['ninth']), 1, \
                        np.where((IN_SAMPLE['Bull-Bear Spread'] > IN_SAMPLE['ninth']), 0,1))

IN_SAMPLE['Portfolio Return'] = IN_SAMPLE['position'] * IN_SAMPLE['Market Return'].shift(1)

IN_SAMPLE['Portfolio Index'] = (1+IN_SAMPLE['Portfolio Return']).cumprod()

IN_SAMPLE['Market Index'] = (1+IN_SAMPLE['Market Return']).cumprod()

IN_SAMPLE[['Market Index','Portfolio Index']].plot(logy=True, title="Great Recession to COVID-19")
#%%

#You need to add the Risk Free Rate and Market Equity Premium
market_data['Market'] = market_data['Mkt-RF']+market_data['RF']

market_data = market_data/100
market_data['Market Index'] = (1+market_data['Market']).cumprod()
market_data['Equity Premium Index'] = (1+market_data['Mkt-RF']).cumprod()
#%%

#####Testing Statistical Significance of Equity Risk Premium#########

#Sample Size
N = market_data['Mkt-RF'].shape[0]

ZEROS = [0] * N
## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(market_data['Mkt-RF'], ZEROS, nan_policy='omit')
print("t = " + str(t2))
print("p = " + str(p2/2)) #one sided t test
#%%

IN_SAMPLE['Rolling_Allocation'] = IN_SAMPLE['position'].rolling(26).mean()
IN_SAMPLE['Rolling_Allocation'].plot(title="Rolling 6 Month Average Allocation")
IN_SAMPLE['Rolling_6mo_Spread'] = IN_SAMPLE['Bull-Bear Spread'].rolling(26).mean()
IN_SAMPLE[['Bull-Bear Spread','Rolling_6mo_Spread']].plot(title="Bull-Bear Spread")
IN_SAMPLE[['Bull-Bear Spread','Rolling_6mo_Spread']].plot(title="Bull-Bear Spread")

IN_SAMPLE[['Market Index','Portfolio Index']].plot(logy=True, title="Timing is Tough")
#%%
##### For Loop through all Iterations to see if could have outperformed####

from itertools import product

df2 = pd.DataFrame()
for i in np.arange(0.0, 1.01, 0.01):
    df2[str(np.round(i,2))+'_percentile'] = AAII_data['Bull-Bear Spread'].expanding(4).quantile(i, interpolation='lower')

df2['Bull-Bear Spread'] = AAII_data['Bull-Bear Spread']
new_df2=pd.merge(df2,merge['Market Return'], how='inner', left_index=True, right_index=True)
new_df2['Market Index'] = (1+new_df2['Market Return']).cumprod()
new_df2['Market Log Return'] = np.log(new_df2['Market Index'] / new_df2['Market Index'].shift(1))

Flat1 = np.arange(0.0, 1.01, 0.01)
Leverage1 = np.arange(0.0, 1.01, 0.01)

results = pd.DataFrame()
for i, j in product(Flat1, Leverage1):
    data = pd.DataFrame(new_df2)
    data.dropna(inplace = True)
    data['position'] = np.where((data['Bull-Bear Spread'] > data[str(np.round(i,2))+'_percentile']), 0,
                        np.where((data['Bull-Bear Spread'] < data[str(np.round(j,2))+'_percentile']),2,1))
    data['Portfolio Return'] = data['position'] * data['Market Log Return'].shift(1)
    data['Portfolio Index'] = (1+data['Portfolio Return']).cumprod()
    data.dropna(inplace = True)
    D = len(data)
    perf = data[['Market Return', 'Portfolio Return']].add(1).prod() ** ((252/5) / D) - 1 
    std = data[['Market Return', 'Portfolio Return']].std() * (252/5) ** 0.5
    Sharpe = perf/std
    results = results.append(pd.DataFrame(
                {'Flat Percentile': np.round(i,2), '2x Percentile': np.round(j,2),
                 'MARKET SHARPE': Sharpe['Market Return'],
                 'STRATEGY SHARPE': Sharpe['Portfolio Return'],
                 'MARKET STD': std['Market Return'],
                 'STRATEGY STD': std['Portfolio Return'],
                 'MARKET RETURN': perf['Market Return'],
                 'STRATEGY RETURN': perf['Portfolio Return'],
                 'OUT': Sharpe['Portfolio Return'] - Sharpe['Market Return']},
                 index = [0]), ignore_index = True)

results.sort_values('OUT', ascending = False).head(15)

heatmapresults = results[["Flat Percentile","2x Percentile","OUT"]]
heat_map = sns.heatmap(heatmapresults)
outperformanceresults = heatmapresults.pivot("Flat Percentile","2x Percentile","OUT")
ax = sns.heatmap(outperformanceresults, annot = False,
                 cmap="Reds").set_title('Strategy Sharpe Relative to Market Sharpe')
#%%

