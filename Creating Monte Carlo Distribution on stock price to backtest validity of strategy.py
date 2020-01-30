#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:24:06 2020

@author: downey
"""


# Importing necessary libraries
import pandas as pd
import numpy as np
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

'''Create a Mean Reversion Strategy with Bollinger Bands

Compute the Bollinger bands using the 30-day window for rolling mean and 
rolling standard deviation. Use 1 standard deviation for the purpose.
Buy when the price hits the lower Bollinger band and sell when the price 
hits the upper Bollinger band.'''

Security = web.DataReader("GE", "av-daily-adjusted", start, end, access_key = 'apikeyhere')                                   
data = Security.iloc[:,0:6]

# Defining parameters
window = 30
sd = 1

#create adjusted Open High Low from Adjusted Close Price
data['Adj. Open'] = (data['adjusted close']/data['close']) * data['open']
data['Adj. High'] = (data['adjusted close']/data['close']) * data['high']
data['Adj. Low'] = (data['adjusted close']/data['close']) * data['low']


data.head()
# Computing moving average and standard deviation
data['moving_average'] = data['adjusted close'].rolling(window=window, center=False).mean()
data['std_dev'] = data['adjusted close'].rolling(window=window, center=False).std()

# Computing upper and lower bands
data['upper_band'] = data['moving_average'] + sd * data['std_dev']
data['lower_band'] = data['moving_average'] - sd * data['std_dev']

# Computing signals
data['long_entry'] = data['adjusted close'] < data.lower_band   
data['long_exit'] = data['adjusted close'] >= data.moving_average

data['positions_long'] = np.nan  
data.loc[data.long_entry,'positions_long'] = 1  
data.loc[data.long_exit,'positions_long'] = 0 

data.positions_long = data.positions_long.fillna(method='ffill')

data['short_entry'] = data['adjusted close'] > data.upper_band   
data['short_exit'] = data['adjusted close'] <= data.moving_average

data['positions_short'] = np.nan  
data.loc[data.short_entry,'positions_short'] = -1  
data.loc[data.short_exit,'positions_short'] = 0  

data.positions_short = data.positions_short.fillna(method='ffill')  

data['positions'] = data.positions_long + data.positions_short

data.head()

data.tail()

data['Security Returns'] = data['adjusted close'].pct_change()
data['Strategy Returns'] = data['positions'].shift(1) * data['Security Returns']

portfolio_index = (1 + data[['Security Returns','Strategy Returns']]).cumprod()
portfolio_index.plot()


from scipy.stats import t
from scipy.stats import norm

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

df = 2.477
mean, var, skew, kurt = t.stats(df, moments='mvsk')

x = np.linspace(t.ppf(0.01, df), t.ppf(0.99, df), 100)
ax.plot(x, t.pdf(x, df),'r-', lw=5, alpha=0.6, label='t pdf')

rv = t(df)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')


data2 = data['Security Returns'].dropna()
df3 = t.fit(data2)
df3

#create 100 random price series with similar T distribution as the data
t_distr_random_price_series = t.rvs(df, loc = df3[1], scale = df3[2], size=(5020,100))

t_distr_random_price_series


#Create two series of same length as 
df1 = norm.fit(data2)
normal_random_series = norm.rvs(size = (5020,2), loc = df1[0], scale = df1[1])

dates = data2.index

Monte_Carlo_DataFrame = pd.DataFrame(t_distr_random_price_series, index=dates)

'''We have monte carlo dataframe but only for daily returns.
To create an artificial OHLC data, I am wondering of using the MC returns to create
The artificial Close prices differences, and then creating from their artificial
Open using norm random from previous close based on mean and standard deviation,
and then High and Low based on norm rand from with sample mean and stand deviation
in relation to Opening price?
'''
Monte_Carlo_DataFrame.head()

#Index plot of Monte Carlo Returns
MC_Index = (1 + Monte_Carlo_DataFrame).cumprod()
MC_Index.plot()
