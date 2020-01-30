#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:16:11 2019

@author: downey
"""
import pyfolio 
import numpy as np
import pandas as pd
from pylab import mpl, plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

pd.set_option('display.max_columns',110)
pd.set_option('display.max_rows',1000)
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

raw = pd.read_csv('/FF_daily.csv', index_col = 0, parse_dates = True)

#Fama French Market US Equity Returns 1926 to mid 2019

raw.tail()
raw.head()
raw.columns = ['Market.Close']
raw.info()

symbol = 'Market.Close'

data = (
    pd.DataFrame(raw[symbol])
    .dropna()
  )

'''
Backtest a strategy using three moving averages on any indices such as 
Nifty50, SPY, HSI
and so on
1. Compute three moving averages of 20, 40, and 80
2. Go long when the price crosses above all three moving averages
3. Exit the long position when the price crosses below any of the three moving averages
4. Go short when the price crosses below all three moving averages
5. Exit the short position when the price crosses above any of the three moving
averages
'''

results = pd.DataFrame()
data = pd.DataFrame(raw[symbol])
data.dropna(inplace = True)
data['Returns'] = data[symbol].pct_change()
data['SMA1'] = data[symbol].rolling(20).mean()
data['SMA2'] = data[symbol].rolling(40).mean()
data['SMA3'] = data[symbol].rolling(80).mean()
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


#1. Compute the buy and hold returns
print("Buy and hold returns annualized " + str(round(perf[0], 4)))

#2. Compute the strategy returns and compare it with the buy and hold returns
print("strategy returns annualized " + str(round(perf[1], 4)) + \
      " resulting in outperformance of " + str(round(perf[1]-perf[0], 4)))

#3. Plot buy and hold returns and strategy returns in a single chart
portfolio_index = (1 + data[['Strategy','Returns']]).cumprod()
portfolio_index.plot()

#4. Used log scale to be more readable
import matplotlib.pyplot as plt
plt.plot(portfolio_index)
plt.yscale('log')
plt.show()


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
#a. Number of positive trades
portfolio_index.columns = ['Strategy Index', 'Buy and Hold Index']
data = data.join(portfolio_index, how = "left")

data['Signal'] = np.where((data['Position'] == -1), 'Sell', \
        np.where((data['Position'] == 1), 'Buy', ''))

data['Signal Price'] = np.where((data['Signal'] == 'Sell') | (data['Signal'] == 'Buy'), \
    data['Strategy Index'], 0)

data['Status'] = np.where((data['Signal'].shift(1) == 'Sell'), 'Short', \
        np.where((data['Signal'].shift(1) == 'Buy'), 'Long', 'Flat'))

data['Entry Price'] = np.nan

#fill in first Entry Price, which is needed for the next for loop
data.iloc[0, 12] = data.iloc[0, 10]

#Entry Price Logic
#If the current status is the same as previous status, carry forward Entry price
#otherwise if Short/Long use current Signal Price
for i in range(1, len(data)):
    data.iloc[i, 12] = np.where((data.iloc[i-1,11] == data.iloc[i,11]), data.iloc[i-1,12], \
        np.where((data.iloc[i,11] == 'Short') | (data.iloc[i,11] == 'Long'), data.iloc[i,0], 0))

data['P & L'] = np.where(((data['Status'] == 'Flat') & \
        (data['Status'].shift(1) == 'Short')), \
        (data['Entry Price'].shift(1) -  data['Market.Close']), \
        np.where(((data['Status'] == 'Flat') & \
        (data['Status'].shift(1) == 'Long')), \
        (data['Market.Close'] - data['Entry Price'].shift(1)), np.NaN))


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

print('The Average profit/loss per trade ' + str(np.round(Avg_PL_per_trade,4)) + \
      ', but since the index has grown so much over time this statistic is not extremely meaningful')

#f. Hit Ratio
Hit_Ratio = positive_trades / total_trades
print('The Hit Ratio for this strategy is ' + str(np.round(Hit_Ratio,4)))
#g. Highest profit & loss in a single trade

trade_data = data['P & L'].dropna()
Max_Profit = trade_data.max()
Max_Loss = trade_data.min()
print('The Highest Profit on any one trade was ' + str(np.round(Max_Profit,2)) + ' dollars')
print('The Highest Loss on any one trade was ' + str(np.round(Max_Loss,2)) + ' dollars')


pyfolio.create_simple_tear_sheet(data['Strategy'])
pyfolio.create_full_tear_sheet(data['Strategy'])


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

#####Testing Statistical Significance of Alpha#########

from scipy import stats
data.head()
#Sample Size
N = data.shape[0]

#Calculate the variance to get the standard deviation

#For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
var_alpha = data['Strategy'].var(ddof=1)

## Calculate the t-statistics
t = (data['Strategy'].mean() - data['Returns'].mean()) / np.sqrt(var_alpha/N)


## Compare with the critical t-value
#Degrees of freedom
df = N-1

#p-value after comparison with the t 
p = 1 - stats.t.cdf(t,df=df)

print("t = " + str(t))
print("p = " + str(p))
### You can see that after comparing the  

ZEROS = [0] * N
## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(data['Strategy'],data['Returns'])
print("t = " + str(t2))
print("p = " + str(p2/2)) #one sided t test
