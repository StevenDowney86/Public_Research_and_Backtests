#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:05:35 2019

@author: downey
"""

#this is a vol targeted strategy to see how volatility targeting works in 
#in relation to a static buy and hold portfolio. I used a simple average of 
#rolling volatility. The next step would be to use EWMA or GARCH models and
#then probably blend them as an ensemble method

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

#importing the Fama French historical US stock market data
#can also get from 
#https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html
raw = pd.read_csv('/Users/downey/Coding/Python/Python for Finance/F-F_Research_Data_Factors_daily.csv', index_col = 0, parse_dates = True)
raw.head()
#drop two columns SMB = Small minus Big market cap, and HML = Value factor (low
#price to book minus high price to book) 
raw = raw.drop(columns=['SMB', 'HML'])
#combine the Market Risk Premium and Risk Free rate to get market return
raw['MKT+RF'] = raw['Mkt-RF']+raw['RF']
#divide values by 100 to get decimal percentage 
raw = raw/100

#create blended rolling vol time frames to minimize parameter specification 
#risk. We are trying to get the general signal vs. exactly right. 
#Using 20 - 140 trading days as the short term rolling window
rolling_vol = pd.DataFrame()
rolling_vol['rolling sd 20'] = raw['MKT+RF'].rolling(20).std()*252**.5
rolling_vol['rolling sd 40'] = raw['MKT+RF'].rolling(40).std()*252**.5
rolling_vol['rolling sd 60'] = raw['MKT+RF'].rolling(60).std()*252**.5
rolling_vol['rolling sd 80'] = raw['MKT+RF'].rolling(80).std()*252**.5
rolling_vol['rolling sd 100'] = raw['MKT+RF'].rolling(100).std()*252**.5
rolling_vol['rolling sd 120'] = raw['MKT+RF'].rolling(120).std()*252**.5
rolling_vol['rolling sd 140'] = raw['MKT+RF'].rolling(140).std()*252**.5

rolling_vol.tail()

rolling_vol['average'] = rolling_vol.mean(axis = 1)
rolling_vol['average'].describe()


#create lagged return series to invest in to eliminate look ahead bias.
#We can only invest in the close following the signal day.
raw.tail()
raw['MKT+RF.lag'] = raw['MKT+RF'].shift(-1)
raw['RF.lag'] = raw['RF'].shift(-1)

#merge the equity vol dataframe with the master dataframe
raw['rolling vol'] = rolling_vol['average']

#Choose a target vol level, 16% is the historical average equity vol, not 
#that you would have known that at the time but you can choose a level
raw['target vol'] = .16

#construct the equity weight as a relation to target vol
raw['equity_weight'] = raw['target vol']/raw['rolling vol']

#Cash weight will be the opposite
raw['cash weight'] = 1 - raw['equity_weight']

#get rid of NAs
raw = raw.dropna()
raw.tail()

#compute the before fee portfolio return
raw['portfolio_return'] = (raw['MKT+RF.lag'] * raw['equity_weight']) + (raw['RF.lag'] * raw['cash weight'])
raw.tail()

#annualized return
D = len(raw)
raw['portfolio_return'].add(1).prod() ** (252 / D) - 1

#annualized standard deviation
np.sqrt(252*raw['portfolio_return'].var())

#chose transaction fee of 15 bps, though Andrew Lo used 5bps in his simulation
#but he estimates using futures the one way cost could be about 1 bps.
#However, transaction fees were about 2% before May day when commissions became
#unfixed, so you would be able to execute this strategy since 1926, but you can going
#forward.

Transaction_Fee = 0.0015

#calculate the portfolio turnover to calculate the fee
Equity_weight_change = raw['equity_weight'].shift(-1) - raw['equity_weight']
Cash_weight_change = raw['cash weight'].shift(-1) - raw['cash weight']

raw['Equity_weight_change'] = Equity_weight_change
raw['Cash_weight_change'] = Cash_weight_change

Portfolio_Turnover = abs(raw['Equity_weight_change']) + abs(raw['Cash_weight_change'])

raw['Portfolio Turnover'] = Portfolio_Turnover

raw['Transaction_Fees'] = raw['Portfolio Turnover'] * Transaction_Fee

raw['portfolio_return_after_fees'] = raw['portfolio_return'] - raw['Transaction_Fees']

#annualized return
D = len(raw)
raw['portfolio_return_after_fees'].add(1).prod() ** (252 / D) - 1

#annualized standard deviation
np.sqrt(252*raw['portfolio_return_after_fees'].var())

#See if the actual porfolio vol came close to the target using 100 day, 252 days, 756 days (3 year)
portfolio_volatility = pd.DataFrame()
portfolio_volatility['100 day rolling vol'] = raw['portfolio_return_after_fees'].rolling(100).std()*252**.5
portfolio_volatility['252 day rolling vol'] = raw['portfolio_return_after_fees'].rolling(252).std()*252**.5
portfolio_volatility['756 day rolling vol'] = raw['portfolio_return_after_fees'].rolling(756).std()*252**.5

#You can see using a longer measurement window it does a decent job at staying around the target
#volatility level

portfolio_volatility.describe()

portfolio_volatility.plot()

#annualized return for all portfolios
D = len(raw)
raw['portfolio_return'].add(1).prod() ** (252 / D) - 1
raw['MKT+RF'].add(1).prod() ** (252 / D) - 1
raw['portfolio_return_after_fees'].add(1).prod() ** (252 / D) - 1

#Calculate maxdrawdown

#create total wealth index of the portfolio

Portfolio_Wealth = pd.DataFrame((1 + raw['portfolio_return_after_fees']).cumprod())
Portfolio_Wealth['Market Return'] = (1 + raw['MKT+RF']).cumprod()
Portfolio_Wealth['Portfolio_return'] = (1 + raw['portfolio_return']).cumprod()

Portfolio_Wealth.tail()

# We are going to use a trailing 252 trading day window
window = 252

# Calculate the max drawdown in the past window days for each day in the series.
# Use min_periods=1 if you want to let the first 252 days data have an expanding window

Roll_Max = Portfolio_Wealth['Market Return'].rolling(1000).max()
Daily_Drawdown = Portfolio_Wealth['Market Return']/Roll_Max - 1.0

# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window
Max_Daily_Drawdown = Daily_Drawdown.rolling(1000).min()

# Plot the results
Daily_Drawdown.plot()
Max_Daily_Drawdown.plot()

#Maximum Drawdown
Max_Daily_Drawdown.dropna().min()

# We are going to use a trailing 252 trading day window
window = 252

# Calculate the max drawdown in the past window days for each day in the series.
# Use min_periods=1 if you want to let the first 252 days data have an expanding window

Roll_Max_port = Portfolio_Wealth['portfolio_return_after_fees'].rolling(1000).max()
Daily_Drawdown_port = Portfolio_Wealth['portfolio_return_after_fees']/Roll_Max_port - 1.0

# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window
Max_Daily_Drawdown_port = Daily_Drawdown_port.rolling(1000).min()

# Plot the results
Daily_Drawdown_port.plot()
Max_Daily_Drawdown_port.plot()

#Maximum Drawdown
Max_Daily_Drawdown_port.dropna().min()




