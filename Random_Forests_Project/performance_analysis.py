#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:55:27 2019

@author: downey
"""
import pandas as pd
import numpy as np

##### Performance and Risk Analysis Functions ####

def annualized_return(x):
    '''Compute Annaulzied Return'''
    gross_return = x.iloc[-1]/x.iloc[0]
    shape = x.shape
    days = shape[0]
    years = days/252
    ann_return = gross_return ** (1/years)
    ann_return = ann_return - 1
    df = pd.DataFrame({'Portfolio':ann_return.index, \
                       'Annualized Return':ann_return.values})
    return df

def annualized_standard_deviation(x):
    '''Compute Annualized Standard Deviation'''
    data2 = x.pct_change()
    std = data2.std() * 252 ** 0.5
    df = pd.DataFrame({'Portfolio':std.index, 'Standard Deviation':std.values})
    return df

def max_drawdown(x):
    '''Max Peak to Trough Loss'''
    roll_max = x.expanding().max()
    daily_drawdown = x/roll_max - 1.0
    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    max_daily_drawdown = daily_drawdown.expanding().min()
    # Plot the results
    #Daily_Drawdown.plot()
    #Max_Daily_Drawdown.plot()
    max_dd = np.min(max_daily_drawdown)
    max_dd = pd.DataFrame({'Portfolio':max_dd.index, \
                                 'Max Drawdown':max_dd.values})
    return max_dd

def gain_to_pain_ratio(x):
    '''Calculate Schwager's Gain to Pain Ratio'''
    returns = x.pct_change().dropna()
    positive_returns = returns[returns >= 0].sum()
    negative_returns = abs(returns[returns < 0].sum())
    gain_to_pain = positive_returns / negative_returns
    gain_to_pain = pd.DataFrame({'Portfolio':gain_to_pain.index, \
                                 'Gain to Pain Ratio':gain_to_pain.values})
    return gain_to_pain

def calmar_ratio(x):
    '''Annualized Return over Max Drawdown'''
    calmar = annualized_return(x).iloc[:, 1]/-max_drawdown(x).iloc[:, 1]
    calmar = pd.DataFrame({'Portfolio':x.columns, 'Calmar Ratio':calmar.values})
    return calmar

def sharpe_ratio(x, RF=0):
    '''Annualized Return - RF rate / Standand Deviation'''
    returns = annualized_return(x)
    std = annualized_standard_deviation(x)
    data = returns.merge(std)
    data['Sharpe Ratio (RF = ' + str(RF) + ')'] = \
        (data['Annualized Return']-float(RF))/data['Standard Deviation']
    sharpe = data[['Portfolio', 'Sharpe Ratio (RF = ' + str(RF) + ')']]
    return sharpe

def sortino_ratio(x, RF=0):
    '''Similar to Sharpe Ratio but denominator is Std Dev. of downside vol'''
    returns = annualized_return(x)
    RF_daily = RF/252
    returns_data = x.pct_change().dropna()
    downside_excess_returns = returns_data[(returns_data - RF_daily) > 0]
    std = downside_excess_returns.std() * 252 ** 0.5
    df = pd.DataFrame({'Portfolio':std.index, 'Downside Standard Deviation':std.values})
    data = returns.merge(df)
    data['Sortino Ratio (RF = ' + str(RF) + ')'] = \
        (data['Annualized Return']-float(RF))/data['Downside Standard Deviation']
    sortino = data[['Portfolio', 'Sortino Ratio (RF = ' + str(RF) + ')']]
    return sortino
