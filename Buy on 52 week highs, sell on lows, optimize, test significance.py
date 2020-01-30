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
Backtest a strategy where you buy on the dips and sell the next day
1. Buy when at 52 week high
5. Sell on 52 week low
'''


data = pd.DataFrame(raw[symbol])
data.dropna(inplace = True)
data['Returns'] = data[symbol].pct_change()
data.dropna(inplace = True)

data['52 week high'] = data['Market.Close'].rolling(252).max()
data['52 week low'] = data['Market.Close'].rolling(252).min()

data.dropna(inplace = True)

data.head()
data.tail()

data['Signal'] = ''
data.iloc[0,4] = 'Flat'

for i in range(1, len(data)):
    data.iloc[i, 4] = np.where(data.iloc[i,0] > data.iloc[i-1,2], 'Long',\
             np.where(data.iloc[i,0] < data.iloc[i-1,3], 'Sell', \
             np.where(data.iloc[i-1,4] == 'Long', 'Long','Flat')))

data['Position'] = np.where(data['Signal'] == 'Long',1,0)
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

pyfolio.create_simple_tear_sheet(data['Strategy'])
pyfolio.create_full_tear_sheet(data['Strategy'])


##############Optional - Optimize the Strategy#################################

import time

t0 = time.time()

Day_High = range(90, 300, 10)

results = pd.DataFrame()
for dayhigh in Day_High:
   data = pd.DataFrame(raw)
   data.dropna(inplace = True)
   data['Returns'] = data.pct_change()
   data.dropna(inplace = True)
   data['52 week high'] = data['Market.Close'].rolling(dayhigh).max()
   data['52 week low'] = data['Market.Close'].rolling(dayhigh).min()
   data.dropna(inplace = True)
   data['Signal'] = ''
   data.iloc[0,4] = 'Flat'
   for i in range(1, len(data)):
       data.iloc[i, 4] = np.where(data.iloc[i,0] > data.iloc[i-1,2], 'Long',\
             np.where(data.iloc[i,0] < data.iloc[i-1,3], 'Sell', \
             np.where(data.iloc[i-1,4] == 'Long', 'Long','Flat')))
   data['Position'] = np.where(data['Signal'] == 'Long',1,0)
   data['Strategy'] = data['Position'].shift(1) * data['Returns']
   data.dropna(inplace = True)
   D = len(data)
   perf = data[['Returns', 'Strategy']].add(1).prod() ** (252 / D) - 1
   portfolio_index = (1 + data[['Returns','Strategy']]).cumprod()
   Max_DD = Max_Drawdown(portfolio_index)
   std = data[['Returns', 'Strategy']].std() * 252 ** 0.5
   Sharpe = perf/std
   results = results.append(pd.DataFrame(
                {'Days High': dayhigh,
                 'MARKET Returns': perf['Returns'],
                 'STRATEGY Returns': perf['Strategy'],
                 'Market Sharpe': Sharpe[0],
                 'Strategy Sharpe': Sharpe[1],
                 'Market Max DD': Max_DD[0],
                 'Strategy Max DD': Max_DD[1],
                 'OUT': Sharpe['Strategy'] - Sharpe['Returns']},
                 index = [0]), ignore_index = True)


results.sort_values('STRATEGY Returns', ascending = False).head(15)
results
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
