#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:06:49 2020

@author: downey
"""

import numpy as np
import pandas as pd
from pylab import mpl, plt
import talib

#import pandas_datareader as pdr
from datetime import datetime
import pandas_datareader.data as web

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

pd.set_option('display.max_columns',110)
pd.set_option('display.max_rows',1000)
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

start=datetime(2000, 1, 1)
end=datetime(2019, 12, 31)

Security = web.DataReader("F", "av-daily-adjusted", start, end, access_key = 'apikeyhere')                                   
raw = Security.iloc[:,0:6]
symbol = 'adjusted close'

raw['RSI'] = talib.RSI(raw['adjusted close'], timeperiod=14)


'''
Strategy based on RSI indicator
1. Buy the instrument such as Nifty or SPY when the RSI is less than 35 (CHANGED TO 35 SINCE NO DATA BELOW 15)
2. Exit conditions:
a. Take profit of 5% or RSI > 75
b. Stop loss of - 2%
3. Optional: Optimise the strategy by adjusting the RSI value. Also, take profit and stop
loss criteria can be different for each stock
4. Note: You can use TA-Lib in Python to compute the RSI value
'''

data = raw.copy()
data.dropna(inplace = True)
data['Returns'] = data[symbol].pct_change()
data.dropna(inplace = True)

data.head(5)
data.tail(5)
data.shape
data[data['RSI'] < 35]


data['Signal'] = ''
data.iloc[0,8] = 'Flat'

for i in range(1, len(data)):
    data.iloc[i, 8] = np.where(data.iloc[i,6] < 35, 'Buy',\
        np.where(data.iloc[i,6] > 75, 'Sell','Flat'))

data['Signal Price'] = np.where((data['Signal'] == 'Buy'), \
    data['adjusted close'], 0)


#fill in first Entry Price, which is needed for the next for loop
data['Entry Price'] = np.NaN
data.iloc[0, 10] = 0
data.iloc[1, 10] = 0

data['Stop Loss Price'] = np.NaN
data['Take Profit Price'] = np.NaN
data['Status'] = ''
data.iloc[0,13] = 'Flat'

#Work through stop loss price, entry price, take profit price, and status
for i in range(1, len(data)-1):
    #below is the status logic for the 'Status' Column
    data.iloc[i,13] = np.where((data.iloc[i,8] == 'Buy') & (data.iloc[i-1,8] == 'Flat'), 'Long',        
      np.where((data.iloc[i-1,13] == 'Long') & (data.iloc[i,4] > data.iloc[i-1,12]), 'TP',
        np.where((data.iloc[i-1,13] == 'Long') & (data.iloc[i,8]  == 'Sell'), 'Long Exit',
           np.where((data.iloc[i-1,13] == 'Long') & (data.iloc[i,4] < data.iloc[i-1,11]), 'SL', 
                    np.where(data.iloc[i-1,13] == 'Long', 'Long','Flat'
                   )))))
    #This is the Entry Price Logic
    data.iloc[i, 10] = np.where((data.iloc[i-1,13] == data.iloc[i,13]), data.iloc[i-1,10], \
             np.where((data.iloc[i-1,13] == 'Flat') & (data.iloc[i,13] == 'Long'), data.iloc[i+1,4], 0))
    #this is the SL price
    data.iloc[i,11] = .98 * data.iloc[i,10]
    #this is the TP price
    data.iloc[i,12] = 1.05 * data.iloc[i,10]
      
data['P & L'] = np.where((((data['Status'] == 'TP') | (data['Status'] == 'SL')) & (data['Status'].shift(1) == 'Long')), \
    data['adjusted close'].shift(-1)-data['Entry Price'].shift(1), np.NaN)

data['Position'] = np.where(data['Status'] == 'Long', 1, 0)
data['Strategy'] = data['Position'].shift(1) * data['Returns']

Days = len(data)
perf = data[['Returns', 'Strategy']].add(1).prod() ** (252 / Days) - 1 
perf

#1. Compute the buy and hold returns
print("Buy and hold returns annualized " + str(round(perf[0], 4)))

#2. Compute the strategy returns and compare it with the buy and hold returns
print("strategy returns annualized " + str(round(perf[1], 4)) + \
      " resulting in outperformance of " + str(round(perf[1]-perf[0], 4)))

#3. Plot buy and hold returns and strategy returns in a single chart
portfolio_index = (1 + data[['Strategy','Returns']]).cumprod()
portfolio_index.plot()

#5. Compute the Sharpe ratio
std = data[['Returns', 'Strategy']].std() * 252 ** 0.5
Sharpe = perf/std
print("The Strategy Sharpe Ratio is " + str(round(Sharpe[1], 4)))

#6. Compute and plot the drawdown of the strategy
def Max_Drawdown_Chart(x):
    # We are going to use a trailing 252 trading day window
    Roll_Max = x.expanding().max()
    Daily_Drawdown = x/Roll_Max - 1.0
    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = Daily_Drawdown.expanding().min()
    # Plot the results
    Daily_Drawdown.plot()
    Max_Daily_Drawdown.plot()
    
def Max_Drawdown(x):
    Roll_Max = x.expanding().max()
    Daily_Drawdown = x/Roll_Max - 1.0
    return Daily_Drawdown.min()

Max_Drawdown_Chart(portfolio_index['Strategy'])
print("Max Drawdown is " + str(Max_Drawdown(portfolio_index['Strategy'])))


#7. Compute the following:
#b. Number of negative trades       
negative_trades = (data[data['P & L'] < 0])
print('The Number of Negative Trades ' + str(len(negative_trades)))

# Number of positive trades
positive_trades = (data[data['P & L'] > 0])
print('The Number of Positive Trades ' + str(len(positive_trades)))

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
Hit_Ratio = len(positive_trades) / total_trades
print('The Hit Ratio for this strategy is ' + str(np.round(Hit_Ratio,4)))
#g. Highest profit & loss in a single trade

trade_data = data['P & L'].dropna()
Max_Profit = trade_data.max()
Max_Loss = trade_data.min()
print('The Highest Profit on any one trade was ' + str(np.round(Max_Profit,2)) + ' dollars')
print('The Highest Loss on any one trade was ' + str(np.round(Max_Loss,2)) + ' dollars')


##############Optional - Optimize the Strategy#################################

import time

t0 = time.time()

#SMA
sma1 = range(1, 40, 3)
sma2 = range(20, 60, 3)
sma3 = range(60, 200, 3)

from itertools import product

results = pd.DataFrame()
for SMA1, SMA2, SMA3 in product(sma1, sma2, sma3):
    data = pd.DataFrame(raw[symbol])
    data.dropna(inplace = True)
    data['Returns'] = data[symbol].pct_change()
    data['SMA1'] = data[symbol].rolling(SMA1).mean()
    data['SMA2'] = data[symbol].rolling(SMA2).mean()
    data['SMA3'] = data[symbol].rolling(SMA3).mean()
    data.dropna(inplace = True)
    data['Position'] = np.where((data['Market.Close'] > data['SMA1']) & \
        (data['Market.Close']> data['SMA2']) & \
        (data['Market.Close']> data['SMA3']), 1, 
        np.where((data['Market.Close'] < data['SMA1']) & \
        (data['Market.Close']< data['SMA2']) & \
        (data['Market.Close']< data['SMA3']), -1, 0))
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data.dropna(inplace = True)
    D = len(data)
    perf = data[['Returns', 'Strategy']].add(1).prod() ** (252 / D) - 1 
    std = data[['Returns', 'Strategy']].std() * 252 ** 0.5
    Sharpe = perf/std
    results = results.append(pd.DataFrame(
                {'SMA1': SMA1, 'SMA2': SMA2, 'SMA3': SMA3,
                 'MARKET': Sharpe['Returns'],
                 'STRATEGY': Sharpe['Strategy'],
                 'OUT': Sharpe['Strategy'] - Sharpe['Returns']},
                 index = [0]), ignore_index = True)

results.info()
results.sort_values('OUT', ascending = False).head(15)

t1 = time.time()

total = t1-t0
print('It took ' + str(np.round(total/60,2)) + ' minutes to run the code')


