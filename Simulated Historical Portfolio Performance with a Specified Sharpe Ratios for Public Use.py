#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:58:28 2020

@author: downey
"""

'''This script creates 10000 random historical portfolios with specified Sharpe 
Ratio to simulate drawdowns and underperformance'''

import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from scipy import stats
import os

plt.style.use('seaborn')
pd.set_option('display.max_columns',110)
pd.set_option('display.max_rows',1000)
warnings.filterwarnings('ignore')

#Download Fama French Data Set to then use date index for simulation

from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web

len(get_available_datasets())

ds = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start = '1920-08-30',)

print(ds['DESCR'])

ds[0].head()
raw = ds[0]
raw = raw.drop(columns=['SMB', 'HML'])
#combine the Market Risk Premium and Risk Free rate to get market return
raw['MKT+RF'] = raw['Mkt-RF']+raw['RF']
#divide values by 100 to get decimal percentage 
raw = raw/100

raw['GSPC.O'] = (1+raw['MKT+RF']).cumprod()
raw.tail()  
symbol = 'GSPC.O'

Real_data = (pd.DataFrame(raw[symbol]).dropna())
Real_data = Real_data.pct_change().dropna()
Real_data.tail()
###################normal distribution################################3

#set the average daily return and daily vol
norm_dis = norm.rvs(loc = .10/252, scale = .1 * (252**2), size = 1000)

#get number of days in data set
Real_data_days = Real_data.shape[0]

#create 1000 random price series with similar norm distribution as the data

number_of_portfolios = 10000
Annual_return = .08
Annualized_volatility = .16
daily_mean_return = Annual_return / 252
daily_volatility = Annualized_volatility / (252**.5)

#create the random portfolios
norm_distr_random_price_series = norm.rvs(loc = daily_mean_return, scale = daily_volatility, size=(Real_data_days,number_of_portfolios))

#extract the date index for the convering array to dataframe
Real_data_dates = Real_data.index

#Create Monte Carlo Dataset based on t distribution data
Monte_Carlo_DataFrame = pd.DataFrame(norm_distr_random_price_series, index=Real_data_dates)

#create index from returns
MC_Index = (1 + Monte_Carlo_DataFrame).cumprod()

#use the Performance_Analysis python file to import functions
os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code, and Results/')
from Performance_Analysis import Annualized_Return
from Performance_Analysis import Max_Drawdown
from Performance_Analysis import Gain_To_Pain_Ratio
from Performance_Analysis import Calmar_Ratio
from Performance_Analysis import Sharpe_Ratio
from Performance_Analysis import Sortino_Ratio
#run the for loop through all the columns of simulated data

#to time the code
t0 = time.time()
results = pd.DataFrame()
for i in range(0,number_of_portfolios,1):        
    data = MC_Index.iloc[:,i:i+1].copy()
    data.dropna(inplace = True)
    cum_perf = data
    Max_DD = Max_Drawdown(cum_perf)
    GP_Ratio = Gain_To_Pain_Ratio(cum_perf)  
    Calmar = Calmar_Ratio(cum_perf)
    Sharpe = Sharpe_Ratio(cum_perf, 0)
    Sortino = Sortino_Ratio(cum_perf, 0)
    days = cum_perf.shape[0]
    Ann_Return = Annualized_Return(cum_perf)
    results = results.append(pd.DataFrame(
                {'Ann. Return': float(Ann_Return.iloc[:,1]),
                 'Max Drawdown': float(Max_DD.iloc[:,1]),
                 'Gain-to-Pain RATIO': float(GP_Ratio.iloc[:,1]),
                 'Calmar RATIO': float(Calmar.iloc[:,1]),
                  'Sharpe RATIO': float(Sharpe.iloc[:,1]),
                  'Sortino RATIO': float(Sortino.iloc[:,1])},
                 index = [0]), ignore_index = True)
t1 = time.time()
total = t1-t0
print('It took ' + str(np.round((total/60),2)) + ' minutes to run') 
results.head()

Bin_Size = 100
Winsorize_Threshold = .0
Years = 2019-1926

#winsorize the data and compress outliers if desired
DD = stats.mstats.winsorize(results.iloc[:,1], limits = Winsorize_Threshold)

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(DD, bins = Bin_Size, rwidth = 3)
plt.title('Max Drawdown of ' + str(number_of_portfolios) + \
          ' Random Portfolio over a ' + str(Years) + ' year horizon with ' + \
          str(Annual_return/Annualized_volatility) + ' Sharpe Ratio')

Returns = stats.mstats.winsorize(results.iloc[:,0], limits = Winsorize_Threshold)

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(Returns, bins = Bin_Size, rwidth = 3)
plt.title('Annualized Returns  '+ str(number_of_portfolios) + \
          ' Random Portfolio over a ' + str(Years) + ' year horizon with ' + \
          str(Annual_return/Annualized_volatility) + ' Sharpe Ratio')

GP_Ratios = stats.mstats.winsorize(results.iloc[:,2], limits = Winsorize_Threshold)

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(GP_Ratios, bins = Bin_Size, rwidth = 3)
plt.title('Gain to Pain Ratios of 1000 Random Portfolio from 1926-2019')

Calmar_Ratios = stats.mstats.winsorize(results.iloc[:,3], limits = Winsorize_Threshold)

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(Calmar_Ratios, bins = Bin_Size, rwidth = 3)
plt.title('Calmar Ratios of 1000 Random Portfolio from 1926-2019')

Sharpe_Ratios = stats.mstats.winsorize(results.iloc[:,4], limits = Winsorize_Threshold)

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(Sharpe_Ratios, bins = Bin_Size, rwidth = 3)
plt.title('Sharpe Ratios of '  + str(number_of_portfolios) + \
          ' Random Portfolio over a ' + str(Years) + ' year horizon with a true ' + \
          str(Annual_return/Annualized_volatility) + ' Sharpe Ratio')

Sortino_Ratios = stats.mstats.winsorize(results.iloc[:,5], limits = Winsorize_Threshold)

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(Calmar_Ratios, bins = Bin_Size, rwidth = 3)
plt.title('Sortino Ratios of 1000  Random Portfolio from 1926-2019')
