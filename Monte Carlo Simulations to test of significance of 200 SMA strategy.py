#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:58:28 2020

@author: downey
"""

'''This script creates 1000 random historical portfolios with the same
t distribution/ or norm distribution as sample data and runs the strategy
on that simulated data to see if the strategy performs higher than the top 5%
of the simulated data to see if the backtest is reliable'''

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import t
from scipy.stats import norm
import time


warnings.filterwarnings('ignore')

#start=datetime(1900, 1, 1)
#end=datetime(2019, 1, 1)
            
#you can download the csv file for Fama French Factor Returns  
raw = pd.read_csv('/F-F_Research_Data_Factors_daily.csv', index_col = 0, parse_dates = True)
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

Real_data = (pd.DataFrame(raw[symbol]).dropna())


Real_data = Real_data.pct_change().dropna()

#find the degrees of freedom, mean, and standard deviation for t distribution
t_dist_fit = t.fit(Real_data)

#get number of days in data set
Real_data_days = Real_data.shape[0]

#create 1000 random price series with similar T distribution as the data

#to time the code
t0 = time.time()

number_of_portfoios = 1000

#create the random portfolios
t_distr_random_price_series = t.rvs(t_dist_fit[0], loc = t_dist_fit[1], scale = t_dist_fit[2], size=(Real_data_days,number_of_portfoios))

#extract the date index for the convering array to dataframe
Real_data_dates = Real_data.index

#Create Monte Carlo Dataset based on t distribution data
Monte_Carlo_DataFrame = pd.DataFrame(t_distr_random_price_series, index=Real_data_dates)

#create index from returns
MC_Index = (1 + Monte_Carlo_DataFrame).cumprod()

#Choose the Parameter to Test
SMA = 200

#run the for loop through all the columns of simulated data
results = pd.DataFrame()
for i in range(0,number_of_portfoios,1):        
    data = MC_Index.iloc[:,i:i+1].copy()
    data.dropna(inplace = True)
    data.columns = ['Index Price']
    data['Returns'] = (data['Index Price'] / data['Index Price'].shift(1))-1
    data['SMA'] = data['Index Price'].rolling(SMA).mean()
    data.dropna(inplace = True)
    data['Position'] = np.where(data['Index Price'] > data['SMA'], 1, 0)
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data.dropna(inplace = True)
    cum_perf = (1+data[['Returns', 'Strategy']]).cumprod()
    days = cum_perf.shape[0]
    Ann_Return = (cum_perf[-1:] ** (1/(days/252)))-1
    results = results.append(pd.DataFrame(
                {'MARKET': float(Ann_Return['Returns']),
                 'STRATEGY': float(Ann_Return['Strategy']),
                 'OUT': float(Ann_Return['Strategy'] - Ann_Return['Returns'])},
                 index = [0]), ignore_index = True)

results.sort_values('OUT', ascending = False).head(15)

t1 = time.time()
total = t1-t0

print(total)

#run the strategy on the real data

Real_data = (pd.DataFrame(raw[symbol]).dropna())
data = Real_data.copy()
data.dropna(inplace = True)
data.columns = ['Index Price']
data['Returns'] = (data['Index Price'] / data['Index Price'].shift(1))-1
data['SMA'] = data['Index Price'].rolling(SMA).mean()
data.dropna(inplace = True)
data['Position'] = np.where(data['Index Price'] > data['SMA'], 1, 0)
data['Strategy'] = data['Position'].shift(1) * data['Returns']
data.dropna(inplace = True)
cum_perf = (1+data[['Returns', 'Strategy']]).cumprod()
days = cum_perf.shape[0]
Ann_Return = (cum_perf[-1:] ** (1/(days/252)))-1
results = results.append(pd.DataFrame(
                {'MARKET': float(Ann_Return['Returns']),
                 'STRATEGY': float(Ann_Return['Strategy']),
                 'OUT': float(Ann_Return['Strategy'] - Ann_Return['Returns'])},
                 index = [0]), ignore_index = True)

#plot the histogram to see how the real alpha did vs. the simulated alphas
plt.figure(figsize=[10,8])
x = results.iloc[0:-2,2]
y = results.iloc[-1,2]
n, bins, patches = plt.hist([x, y], bins = 100, rwidth = 3)
plt.title('Alpha of 1000 Random Portfolio and Real Data (Green)')

