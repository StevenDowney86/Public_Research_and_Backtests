#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:16:11 2019

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

start=datetime(2000, 1, 1)
end=datetime(2019, 12, 31)

Security = web.DataReader("GE", "av-daily-adjusted", start, end, access_key = 'apikeyhere')                                   
raw = Security.iloc[:,0:6]
symbol = 'adjusted close'

'''
Buy and sell the next day
1. Buy the stock on the fourth day open, if the stock closes down consecutively for
three days
2. Exit on the next day open
3. Optional: Optimise the strategy by exiting the long position on the same day close
Also, you can optimise the number of down days. There are high chances that the
number of down days would be different for each stock
'''

results = pd.DataFrame()
data = raw
data.dropna(inplace = True)
data['Returns'] = data[symbol].pct_change()
data.dropna(inplace = True)
data['Position'] = np.where((data['close'].shift(1) < data['open'].shift(1)) & \
        (data['close'].shift(2) < data['open'].shift(2)) & \
        (data['close'].shift(3) < data['open'].shift(3)), 1, 0)
data['Strategy'] = data['Position'] * ((data['open']/data['open'].shift(-1))-1)
data.dropna(inplace = True)
D = len(data)
perf = data[['Returns', 'Strategy']].add(1).prod() ** (252 / D) - 1 

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


Max_Drawdown_Chart(portfolio_index['Returns'])
print("Max Drawdown is " + str(Max_Drawdown(portfolio_index['Strategy'])))

#7. Compute the following:
#a. Number of positive trades
portfolio_index.columns = ['Strategy Index', 'Buy and Hold Index']
data = data.join(portfolio_index, how = "left")
data.head(10)
data['Signal'] = np.where((data['Position'] == 0), 'Flat', \
        np.where((data['Position'] == 1), 'Buy', ''))

data['Signal Price'] = np.where((data['Signal'] == 'Buy'), \
    data['open'], 0)

data['Status'] = np.where((data['Signal']== 'Buy'), 'Long', 'Flat')

data['Entry Price'] = np.where((data['Status'] == 'Long'), \
    data['open'], 0)

data['P & L'] = np.where(data['Status'] == 'Long',data['open']-data['open'].shift(-1), 0)
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


##############Optional - Optimize the Strategy#################################
raw = Security.iloc[:,0:6]
results = pd.DataFrame()
LookbackDays = range(1,20,1)

for Lookback in LookbackDays:
    data = raw
    data.dropna(inplace = True)
    data['Returns'] = data[symbol].pct_change()
    data.dropna(inplace = True)
    data['Closed Down'] = np.where(data['close']>data['open'], -1, 1)
    data['Position'] = np.where(data['Closed Down'].rolling(Lookback).mean() == -1, 1, 0)
    data['Strategy'] = data['Position'] * ((data['open']/data['open'].shift(-1))-1)
    data.dropna(inplace = True)
    D = len(data)
    perf = data[['Returns', 'Strategy']].add(1).prod() ** (252 / D) - 1 
    std = data[['Returns', 'Strategy']].std() * 252 ** 0.5
    Sharpe = perf/std
    results = results.append(pd.DataFrame(
                {'Lookback': Lookback,
                 'MARKET': Sharpe['Returns'],
                 'STRATEGY': Sharpe['Strategy'],
                 'OUT': Sharpe['Strategy'] - Sharpe['Returns']},
                 index = [0]), ignore_index = True)

results.sort_values('OUT', ascending = False).head(15)


