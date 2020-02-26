#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:55:27 2019

@author: downey
"""
import pandas as pd
import numpy as np

def Annualized_Return(x):
    gross_return = x.iloc[-1]/x.iloc[0]
    shape = x.shape
    days = shape[0]
    years = days/252
    annualized_return = gross_return ** (1/years)
    annualized_return = annualized_return - 1
    df = pd.DataFrame({'Portfolio':annualized_return.index, 'Annualized Return':annualized_return.values})
    return df

def Annualized_Standard_Deviation(x):
    data2 = x.pct_change()    
    std = data2.std() * 252 ** 0.5
    df = pd.DataFrame({'Portfolio':std.index, 'Standard Deviation':std.values})
    return df

def Max_Drawdown(x):
    Roll_Max = x.expanding().max()
    Daily_Drawdown = x/Roll_Max - 1.0
    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = Daily_Drawdown.expanding().min()
    # Plot the results
    #Daily_Drawdown.plot()
    #Max_Daily_Drawdown.plot()
    Max_Drawdown = np.min(Max_Daily_Drawdown)
    Max_Drawdown = pd.DataFrame({'Portfolio':Max_Drawdown.index, 'Max Drawdown':Max_Drawdown.values})
    return Max_Drawdown

def Gain_To_Pain_Ratio(x):
    #Calculate Schwager's Gain to Pain Ratio
    Returns = x.pct_change().dropna()
    Positive_Returns = Returns[Returns >= 0].sum()
    Negative_Returns = abs(Returns[Returns < 0].sum())
    Gain_To_Pain = Positive_Returns / Negative_Returns
    Gain_To_Pain = pd.DataFrame({'Portfolio':Gain_To_Pain.index, 'Gain to Pain Ratio':Gain_To_Pain.values})
    return Gain_To_Pain
    
def Calmar_Ratio(x):
    Calmar = Annualized_Return(x).iloc[:,1]/-Max_Drawdown(x).iloc[:,1]
    Calmar_Ratio = pd.DataFrame({'Portfolio':x.columns, 'Calmar Ratio':Calmar.values})
    return Calmar_Ratio

def Sharpe_Ratio(x, RF):
    Returns = Annualized_Return(x) 
    std =  Annualized_Standard_Deviation(x)
    Data = Returns.merge(std)
    Data['Sharpe Ratio (RF = ' + str(RF) + ')'] = (Data['Annualized Return']-float(RF))/Data['Standard Deviation']
    Sharpe_Ratio =  Data[['Portfolio','Sharpe Ratio (RF = ' + str(RF) + ')']]
    return Sharpe_Ratio        

def Sortino_Ratio(x, RF):
    Returns = Annualized_Return(x)  
    RF_daily = RF/252
    Returns_data = x.pct_change().dropna()
    downside_excess_returns = Returns_data[(Returns_data - RF_daily) > 0]
    std = downside_excess_returns.std() * 252 ** 0.5
    df = pd.DataFrame({'Portfolio':std.index, 'Downside Standard Deviation':std.values})
    Data = Returns.merge(df)
    Data['Sortino Ratio (RF = ' + str(RF) + ')'] = (Data['Annualized Return']-float(RF))/Data['Downside Standard Deviation']
    Sortino_Ratio =  Data[['Portfolio','Sortino Ratio (RF = ' + str(RF) + ')']]
    return Sortino_Ratio

