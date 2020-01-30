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


