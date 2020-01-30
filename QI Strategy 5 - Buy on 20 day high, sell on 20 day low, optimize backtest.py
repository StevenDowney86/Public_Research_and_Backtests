#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:46:01 2020

@author: downey
"""

import numpy as np
import pandas as pd
from pylab import mpl, plt

#import pandas_datareader as pdr
from datetime import datetime
import pandas_datareader.data as web

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

pd.set_option('display.max_columns',110)
pd.set_option('display.max_rows',1000)
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

start=datetime(2007, 1, 1)
end=datetime(2009, 12, 31)

Security = web.DataReader("SPY", "av-daily-adjusted", start, end, access_key = 'apikeyhere')                                   
raw = Security.iloc[:,0:5]


''' Buy on the High and sell when penetrates the low - Trend Following 

Go long when the stock closes above the last 20 day’s high price
2. Square off the long position when the stock goes below the last 20 day’s low price
3. Optional: Optimise the strategy by adjusting the number of periods. You can choose
to have different number of periods for entering the long and exiting the long'''

data = raw
data.dropna(inplace = True)

data['Adj. Open'] = (data['adjusted close']/data['close']) * data['open']
data['Adj. High'] = (data['adjusted close']/data['close']) * data['high']
data['Adj. Low'] = (data['adjusted close']/data['close']) * data['low']

data = data.iloc[:,4:8]
data['Rolling Window max High'] = data['adjusted close'].rolling(20).max()
data['Rolling Window min Low'] = data['adjusted close'].rolling(20).min()

data['Returns'] = data['adjusted close'].pct_change()
data.dropna(inplace = True)

data['Signal'] = np.where((data['adjusted close'] > data['Rolling Window max High'].shift(1)), 'Buy',
    np.where(data['adjusted close'] < data['Rolling Window min Low'].shift(1),'Sell','Flat'))

data['Signal Price'] = np.where((data['Signal'] == 'Sell') | (data['Signal'] == 'Buy'), \
    data['adjusted close'], 0)

data['Status'] = ''
data.iloc[0,-1] = 'Flat'

for i in range(1, len(data)-1):
    data.iloc[i, 9] = np.where((data.iloc[i-1,7] == 'Buy'), 'Long', \
    np.where((data.iloc[i-1,9] == 'Long') & (data.iloc[i,7] != 'Sell'),'Long', \
    np.where((data.iloc[i-1,9] == 'Long') & (data.iloc[i,7] == 'Sell'),'Long Exit','Flat')
    ))

data['Entry Price'] = np.nan

#fill in first Entry Price, which is needed for the next for loop
data.iloc[0, 10] = 0

#Entry Price Logic
#If the current status is the same as previous status, carry forward Entry price
#otherwise if Short/Long use current Signal Price
for i in range(1, len(data)):
    data.iloc[i, 10] = np.where((data.iloc[i-1,9] == data.iloc[i,9]), data.iloc[i-1,10], \
        np.where((data.iloc[i,9] == 'Long'), data.iloc[i,8], 0))

data.head()

data['P & L'] = np.where(data['Status'] == 'Long Exit',data['adjusted close'] - data['Entry Price'].shift(1), np.NaN)

data['Position'] = np.where(data['Status'] == 'Long',1,0)

data['Strategy Returns'] = data['Position'].shift(1) * data['Returns']
Days = len(data)
perf = data[['Returns', 'Strategy Returns']].add(1).prod() ** (252 / Days) - 1 

#1. Compute the buy and hold returns
print("Buy and hold returns annualized " + str(round(perf[0], 4)))

#2. Compute the strategy returns and compare it with the buy and hold returns
print("strategy returns annualized " + str(round(perf[1], 4)) + \
      " resulting in outperformance of " + str(round(perf[1]-perf[0], 4)))

#3. Plot buy and hold returns and strategy returns in a single chart
portfolio_index = (1 + data[['Strategy Returns','Returns']]).cumprod()
portfolio_index.plot()

#5. Compute the Sharpe ratio
std = data[['Returns', 'Strategy Returns']].std() * 252 ** 0.5
Sharpe = perf/std
print("The Strategy Sharpe Ratio is " + str(round(Sharpe[1], 4)))
print("The Buy and Hold Sharpe Ratio is " + str(round(Sharpe[0], 4)))

#b. Number of negative trades       
print('The Number of Negative Trades ' + str(len(data[data['P & L'] < 0])))

# Number of positive trades
positive_trades = len(data[data['P & L'] > 0])
print('The Number of Positive Trades ' + str(positive_trades))

#c. Total number of signals generated
#where the previous signal is different than the current
signals_generated = np.count_nonzero(data['Signal'] != data['Signal'].shift(1))
print('The number of signals generated is ' + \
      str(signals_generated))

#d. Total number of signals traded
total_trades = data['P & L'].dropna().shape[0]

signals_traded = total_trades/signals_generated

print('The Percent of Signals Traded is ' + str(np.round(signals_traded,4)) + \
      ' and the number of signals traded (total trades) ' + str(total_trades))

#e. Average profit/loss per trade
Total_Profit_Loss = sum(data['P & L'].dropna())
Avg_PL_per_trade = Total_Profit_Loss/total_trades

print('The Average profit/loss per trade ' + str(np.round(Avg_PL_per_trade,4)))
#f. Hit Ratio

Hit_Ratio = positive_trades / total_trades
print('The Hit Ratio for this strategy is ' + str(np.round(Hit_Ratio,4)))
#g. Highest profit & loss in a single trade

trade_data = data['P & L'].dropna()
Max_Profit = trade_data.max()
Max_Loss = trade_data.min()
print('The Highest Profit on any one trade was ' + str(np.round(Max_Profit,2)) + ' dollars')
print('The Highest Loss on any one trade was ' + str(np.round(Max_Loss,2)) + ' dollars')

###############################################################################

'''Optional to Optimize the Look back days'''

import time

t0 = time.time()

data2 = raw
data2.dropna(inplace = True)

data2['Adj. Open'] = (data2['adjusted close']/data2['close']) * data2['open']
data2['Adj. High'] = (data2['adjusted close']/data2['close']) * data2['high']
data2['Adj. Low'] = (data2['adjusted close']/data2['close']) * data2['low']

data3 = data2.iloc[:,4:8]
data3.head()

results = pd.DataFrame()
for lookback in range(10,50,1):
    data = pd.DataFrame(data3)
    data['Rolling Window max High'] = data['adjusted close'].rolling(lookback).max()
    data['Rolling Window min Low'] = data['adjusted close'].rolling(lookback).min()
    data['Returns'] = data['adjusted close'].pct_change()
    data.dropna(inplace = True)
    data.head()
    data['Signal'] = np.where((data['adjusted close'] > data['Rolling Window max High'].shift(1)), 'Buy',
        np.where(data['adjusted close'] < data['Rolling Window min Low'].shift(1),'Sell','Flat'))
    data['Signal Price'] = np.where((data['Signal'] == 'Sell') | (data['Signal'] == 'Buy'), \
    data['adjusted close'], 0)
    data['Status'] = ''
    data.iloc[0,-1] = 'Flat'
    for i in range(1, len(data)-1):
            data.iloc[i, 9] = np.where((data.iloc[i-1,7] == 'Buy'), 'Long', \
            np.where((data.iloc[i-1,9] == 'Long') & (data.iloc[i,7] != 'Sell'),'Long', \
            np.where((data.iloc[i-1,9] == 'Long') & (data.iloc[i,7] == 'Sell'),'Long Exit','Flat')
                ))
    data['Entry Price'] = np.nan
    #fill in first Entry Price, which is needed for the next for loop
    data.iloc[0, 10] = 0
    #Entry Price Logic
    #If the current status is the same as previous status, carry forward Entry price
    #otherwise if Short/Long use current Signal Price
    for i in range(1, len(data)):
        data.iloc[i, 10] = np.where((data.iloc[i-1,9] == data.iloc[i,9]), data.iloc[i-1,10], \
        np.where((data.iloc[i,9] == 'Long'), data.iloc[i,8], 0))
    data['P & L'] = np.where(data['Status'] == 'Long Exit',data['adjusted close'] - data['Entry Price'].shift(1), np.NaN)
    data['Position'] = np.where(data['Status'] == 'Long',1,0)
    data['Strategy Returns'] = data['Position'].shift(1) * data['Returns']
    Days = len(data)
    perf = data[['Returns', 'Strategy Returns']].add(1).prod() ** (252 / Days) - 1 
    results = results.append(pd.DataFrame(
                {'Lookback': lookback,
                 'MARKET': perf['Returns'],
                 'STRATEGY': perf['Strategy Returns'],
                 'OUT': perf['Strategy Returns'] - perf['Returns']},
                 index = [0]), ignore_index = True)

results.sort_values('OUT', ascending = False).head(25)

t1 = time.time()

total = t1-t0
print('It took ' + str(np.round(total/60,2)) + ' minutes to run the code')