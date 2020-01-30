#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:37:16 2020

@author: downey
"""

import numpy as np
import pandas as pd
from pylab import plt
from datetime import datetime
import pandas_datareader.data as web
import talib

pd.set_option('display.max_columns',110)
pd.set_option('display.max_rows',1000)
plt.style.use('seaborn')


import time

t0 = time.time()

start=datetime(2001, 1, 1)
end=datetime(2019, 12, 31)

symbol = 'F'

security = web.DataReader(symbol, "av-daily-adjusted", start, end, access_key = 'apikeyhere')

security.head()
security.tail()
data = security.iloc[:,0:5]

#create adjusted Open High Low from Adjusted Close Price
data['Adj. Open'] = (data['adjusted close']/data['close']) * data['open']
data['Adj. High'] = (data['adjusted close']/data['close']) * data['high']
data['Adj. Low'] = (data['adjusted close']/data['close']) * data['low']

raw_data = data.iloc[:,5:8]
raw_data = raw_data.join(data.iloc[:,4:5])
raw_data.rename(columns={'adjusted close':'Adj. Close'}, inplace=True)

raw_data.head()

class FinancialData:
    def __init__(self, symbol):
        self.symbol = symbol
        self.Close = 'Adj. Close'
        self.High = 'Adj. High'
        self.Low = 'Adj. Low'
        self.prepare_data()
    def prepare_data(self):
        self.data = pd.DataFrame(raw_data)
        self.data['returns'] = np.log(self.data['Adj. Close'] / self.data['Adj. Close'].shift(1))
        self.data.dropna(inplace=True)
    def plot_data(self, cols=None):
        if cols is None:
            cols = self.symbol
        self.data[self.Close].plot(figsize=(10, 6), title = self.symbol)
        
fd = FinancialData(symbol)
fd.prepare_data()
fd.plot_data()

'''
Backtesting Base Class
We are going to implement a base class for event-based backtesting with:

__init__
prepare_data (FinancialBase)
plot_data (FinancialBase)
get_date_price
print_balance
print_net_wealth
place_buy_order
place_sell_order
close_out
'''

class BacktestingBase(FinancialData):
    def __init__(self, symbol, amount):
        super(BacktestingBase, self).__init__(symbol)
        self.initial_amount = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
    def get_date_price(self, bar):
        date = str(self.data.index[bar])[:10]
        price = self.data[self.Close].iloc[bar]
        return date, price
    def print_balance(self, bar):
        date, price = self.get_date_price(bar)
        print(f'{date} | current balance = {self.current_balance}')
    def print_net_wealth(self, bar):
        date, price = self.get_date_price(bar)
        net_wealth = self.current_balance + self.units * price
        print(f'{date} | net wealth = {net_wealth}')
    def place_buy_order(self, bar, amount=None, units=None):
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.current_balance -= units * price
        self.units += units
        self.trades += 1
        print(f'{date} | bought {units} units for {price}') 
        self.print_balance(bar)
        self.print_net_wealth(bar)
    def place_sell_order(self, bar, amount=None, units=None):
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.current_balance += units * price
        self.units -= units
        self.trades += 1
        print(f'{date} | sold {units} units for {price}') 
        self.print_balance(bar)
        self.print_net_wealth(bar)     
    def close_out(self, bar):
        date, price = self.get_date_price(bar)
        print(50 * '=')
        print(f'{date} | *** CLOSING OUT ***')
        print(f'{date} | bought/sold {self.units} units for {price}')
        self.current_balance += self.units * price
        self.units -= self.units
        self.print_balance(bar)
        perf = (self.current_balance / self.initial_amount - 1) * 100
        print(f'{date} | net performance [%] = {perf:.4f}')
        print(f'{date} | number of trades [#] = {self.trades}')
        print(50 * '=')


##############################################################################

'''Turtle Strategy

We initiate a new buy trade if the price goes above ‘x’ candles high
We initiate a new sell trade if the price goes below ‘y’ candles low
Exit trade  when:
    –Price goes against us by ‘a’ times ATR (Average True Range)
    –Price goes in our favor by ‘b’ times ATR (Average True Range)
Only take one position at a time. Ignore  new signals if there is an ongoing trade
Fixed position size of 1 ‘quantity’

'''

##############################################################################

class TurtleBacktesterLS(BacktestingBase):
    def prepare_indicators(self):
        self.data['Candle Rolling High'] = self.data[self.High].rolling(self.Candle).max()
        self.data['Candle Rolling Low'] = self.data[self.Low].rolling(self.Candle).min()
        self.data['ATR'] = talib.ATR(np.asarray(self.data[self.High]), \
                 np.asarray(self.data[self.Low]), np.asarray(self.data[self.Close]))
    def Entry_and_Exit_data(self):
        self.Entry_Data = pd.DataFrame()
        self.Entry_Data['Entry Price'] = np.NaN
        self.Entry_Data['Initial Status'] = ''
        self.Entry_Data['Units Opened'] = 0
        self.Exit_Data = pd.DataFrame()
        self.Exit_Data['Exit Price'] = np.NaN
        self.Exit_Data['Final Status'] = ''  
    def backtest_strategy(self, Candle, SL_Threshold, TP_Threshold):
        self.Candle = Candle
        self.SL = SL_Threshold
        self.TP = TP_Threshold
        self.ATR_range = 14
        self.prepare_indicators()
        self.Entry_and_Exit_data()
        self.units = 0
        self.position = 0
        self.data['position'] = np.NaN
        self.data['P&L'] = np.NaN
        self.data['Entry Price'] = np.NaN
        self.trades = 0
        self.Entry_Price = np.NaN
        self.current_balance = self.initial_amount
        for bar in range(self.ATR_range, len(self.data)):
            date, price = self.get_date_price(bar)
            if self.position in [0, 0]:
                if self.data['Adj. High'].iloc[bar] > self.data['Candle Rolling High'].iloc[bar-1]:
                    print(50 * '-')
                    print(f'{date} | *** GOING LONG ***')
                    #self.place_buy_order(bar, units=(1- self.position) * 1)
                    self.place_buy_order(bar, amount=5000)
                    self.position = 1
                    self.Entry_Price = price
                    self.Entry_Data = self.Entry_Data.append(pd.DataFrame(
                            {'Entry Price': self.Entry_Price,
                             'Initial Status': 'Long Entry', 'Units Opened': self.units},
                             index = [0]), ignore_index = True)
                elif self.data['Adj. Low'].iloc[bar] < self.data['Candle Rolling Low'].iloc[bar-1] :
                    print(50 * '-')
                    print(f'{date} | *** GOING SHORT ***')
                    #self.place_sell_order(bar, units=(1 - self.position) * 1)
                    self.place_sell_order(bar, amount=5000)
                    self.position = -1
                    self.Entry_Price = price
                    self.Entry_Data = self.Entry_Data.append(pd.DataFrame(
                            {'Entry Price': self.Entry_Price,
                             'Initial Status': 'Short Entry', 'Units Opened': self.units},
                             index = [0]), ignore_index = True)
            elif self.position in [0, 1]:
                if self.data['Adj. Low'].iloc[bar] < (self.Entry_Price - (self.SL * self.data['ATR'].iloc[bar-1])) :
                    print(50 * '-')
                    print(f'{date} | *** EXITING LONG - SL ***')
                    self.place_sell_order(bar, units=self.units)
                    #self.place_sell_order(bar, units=(1- self.position) * 1)
                    self.position = 0
                    self.Entry_Price = np.NaN
                    self.Exit_Data = self.Exit_Data.append(pd.DataFrame(
                            {'Final Status': 'Long Exit',
                             'Exit Price': self.data['Adj. Close'].iloc[bar]},
                             index = [0]), ignore_index = True)
                elif self.data['Adj. High'].iloc[bar] > (self.Entry_Price + (self.TP * self.data['ATR'].iloc[bar-1])) :
                    print(50 * '-')
                    print(f'{date} | *** EXITING LONG - TP ***')
                    self.place_sell_order(bar, units=self.units)
                    #self.place_sell_order(bar, units=(1- self.position) * 1) 
                    self.position = 0
                    self.Entry_Price = np.NaN
                    self.Exit_Data = self.Exit_Data.append(pd.DataFrame(
                            {'Final Status': 'Long Exit',
                             'Exit Price': self.data['Adj. Close'].iloc[bar]},
                             index = [0]), ignore_index = True)
                    self.position = 0
            elif self.position in [0, -1]:
                if self.data['Adj. High'].iloc[bar] > (self.Entry_Price + (self.SL * self.data['ATR'].iloc[bar-1])) :
                    print(50 * '-')
                    print(f'{date} | *** EXITING SHORT - SL ***')
                    self.place_buy_order(bar, units=(self.units) * -1)
                    #self.place_buy_order(bar, units=(1- self.position) * 1)
                    self.position = 0
                    self.Entry_Price = np.NaN
                    self.Exit_Data = self.Exit_Data.append(pd.DataFrame(
                            {'Final Status': 'Short Exit',
                             'Exit Price': self.data['Adj. Close'].iloc[bar]},
                             index = [0]), ignore_index = True)
                elif self.data['Adj. Low'].iloc[bar] < (self.Entry_Price - (self.TP * self.data['ATR'].iloc[bar-1])) :
                    print(50 * '-')
                    print(f'{date} | *** EXITING SHORT - TP ***')
                    self.place_buy_order(bar, units=(self.units) * -1)
                    #self.place_buy_order(bar, units=(1- self.position) * 1)
                    self.position = 0
                    self.Entry_Price = np.NaN
                    self.Exit_Data = self.Exit_Data.append(pd.DataFrame(
                            {'Final Status': 'Short Exit',
                             'Exit Price': self.data['Adj. Close'].iloc[bar]},
                             index = [0]), ignore_index = True)
            self.data.iloc[bar,8] = self.position #add the position to position column
        self.data['strategy'] = self.data['returns'].shift(-1) * self.data['position']
        self.Final_PL_Data = self.Entry_Data.join(self.Exit_Data)
        self.close_out(bar)
    def plot_results(self):
       title = f'{self.symbol}'
       self.data[['returns','strategy']].cumsum().apply(np.exp).plot(
               figsize=(10,6), title=title)
 
       
Turtles_Strategy = TurtleBacktesterLS(symbol, 10000)

Turtles_Strategy.backtest_strategy(10,2,2)

Turtles_Strategy.current_balance
Turtles_Strategy.initial_amount

Turtles_Strategy.data.tail(2)
Turtles_Strategy.plot_results()

Turtles_Strategy.Final_PL_Data['PL'] = np.where(Turtles_Strategy.Final_PL_Data['Final Status'] == 'Long Exit', 
    (Turtles_Strategy.Final_PL_Data['Exit Price'] - Turtles_Strategy.Final_PL_Data['Entry Price']) \
    *Turtles_Strategy.Final_PL_Data['Units Opened'],
    (Turtles_Strategy.Final_PL_Data['Exit Price'] - Turtles_Strategy.Final_PL_Data['Entry Price']) \
    *Turtles_Strategy.Final_PL_Data['Units Opened'])

Initial_Price = Turtles_Strategy.data['Adj. Close'].iloc[0]
Ending_Price = Turtles_Strategy.data['Adj. Close'].iloc[-1]

BuyHold_Annualized_Returns = (Ending_Price/Initial_Price) ** (1 / (len(raw_data)/252)) - 1

'''Stuck on figuring out how to add to data file the returns of the strategy'''

Strategy_Annualized_Returns = (Turtles_Strategy.current_balance/Turtles_Strategy.initial_amount) ** (1 / (len(raw_data)/252)) - 1

#1. Compute the buy and hold returns
print("Buy and hold returns annualized " + str(round(BuyHold_Annualized_Returns, 4)))

#2. Compute the strategy returns and compare it with the buy and hold returns
print("strategy returns annualized " + str(round(Strategy_Annualized_Returns, 4)) + \
      " resulting in outperformance of " + str(round(Strategy_Annualized_Returns-BuyHold_Annualized_Returns, 4)))

#3. Plot buy and hold returns and strategy returns in a single chart
Turtles_Strategy.plot_results()

#5. Compute the Sharpe ratio
std = Turtles_Strategy.data['strategy'].std() * 252 ** 0.5
Sharpe = Strategy_Annualized_Returns/std
print("The Strategy Sharpe Ratio is " + str(round(Sharpe, 4)))

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


portfolio_index = (1 + Turtles_Strategy.data[['returns','strategy']]).cumprod()

Max_Drawdown_Chart(portfolio_index['strategy'])
print("Max Drawdown is " + str(Max_Drawdown(portfolio_index['strategy'])))

#7. Compute the following:

#a. Number of positive trades
positive_trades = len(Turtles_Strategy.Final_PL_Data[Turtles_Strategy.Final_PL_Data['PL'] > 0])
print('The Number of Positive Trades was ' + str(positive_trades))


#b. Number of negative trades       
negative_trades = len(Turtles_Strategy.Final_PL_Data[Turtles_Strategy.Final_PL_Data['PL'] < 0])
print('The Number of Negative Trades was ' + str((negative_trades)))

#c. Total number of signals generated
'''Having a hard time figuring out how to track this with a for loop event driven
backtest'''

#d. Total number of signals traded

'''need previous answer to question'''


#e. Average profit/loss per trade
Total_Profit_Loss = sum(Turtles_Strategy.Final_PL_Data['PL'].dropna())
Total_Trades = positive_trades + negative_trades
Avg_PL_per_trade = Total_Profit_Loss/Total_Trades

print('The Average profit/loss per trade ' + str(np.round(Avg_PL_per_trade,4)))

#f. Hit Ratio
Hit_Ratio = positive_trades / Total_Trades
print('The Hit Ratio for this strategy is ' + str(np.round(Hit_Ratio,4)))
#g. Highest profit & loss in a single trade

trade_data = Turtles_Strategy.Final_PL_Data['PL'].dropna()
Max_Profit = trade_data.max()
Max_Loss = trade_data.min()
print('The Highest Profit on any one trade was ' + str(np.round(Max_Profit,2)) + ' dollars')
print('The Highest Loss on any one trade was ' + str(np.round(Max_Loss,2)) + ' dollars')


##############Optional - Optimize the Strategy#################################
Candle_Lookback = range(1,5,1)
Stop_Loss_Multiple_ATR = range(1,3,1)
Take_Profit_Multiple_ATR = range(1,3,1)

from itertools import product
raw = raw_data.copy()
results = pd.DataFrame()
Turtles_Strategy = TurtleBacktesterLS(symbol, 10000)
for CL_Lookback, SL_Mult, TP_Mult in product(Candle_Lookback, Stop_Loss_Multiple_ATR, Take_Profit_Multiple_ATR):
    Turtles_Strategy.backtest_strategy(CL_Lookback,SL_Mult,TP_Mult)
    Strategy_Annualized_Returns = (Turtles_Strategy.current_balance/Turtles_Strategy.initial_amount) ** (1 / (len(raw)/252)) - 1
    BuyHold_Annualized_Returns = (Ending_Price/Initial_Price) ** (1 / (len(raw)/252)) - 1
    data = Turtles_Strategy.data.dropna()
    std = Turtles_Strategy.data[['returns','strategy']].std() * 252 ** 0.5
    Sharpe_Strategy = Strategy_Annualized_Returns/std[1] 
    Sharpe_Market = BuyHold_Annualized_Returns/std[0] 
    results = results.append(pd.DataFrame(
                {'Candle Lookback': CL_Lookback, 'Stop Loss ATR Mult': SL_Mult, 
                 'Take Profit ATR Mult': TP_Mult,
                 'MARKET': Sharpe_Market,
                 'STRATEGY': Sharpe_Strategy,
                 'OUT': Sharpe_Strategy-Sharpe_Market},
                 index = [0]), ignore_index = True)

results.sort_values('OUT', ascending = False).head(10)

t1 = time.time()

total = t1-t0
print(str(total))
