#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:30:03 2020

@author: downey
"""
'''This script uses the Equal Risk Contribution Optimization on a Rolling
Basis with Kenneth French Sector Data vs. and Equal Weight approach and loops
over different lookback periods

The first section is gathering the data and looping through lookback periods

The second section is a single run and the third is equal weight'''

from __future__ import division
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv,pinv
from scipy.optimize import minimize
import datetime 
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web
import pandas as pd

pd.set_option('display.max_columns', 50)

len(get_available_datasets())

#you will need to get the 'file name' from the the actual url via the French
#website.

#can also use 10 industry

ds = web.DataReader('49_Industry_Portfolios_daily', 'famafrench', start = '1926-01-01',
                    )
print(ds['DESCR'])

ds[0].head()

data = ds[0]
data = data.replace(-99.99, np.nan) #with the 49 industry -99.99 indicates no values
data = data.dropna()
data = data/100 #convert to percent returns
index = (1+data).cumprod()

index.head()
index.tail()

PRICES = index

PRICES.head()
PRICES.tail()
PRICES = PRICES.dropna()

'''from https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/'''

 # risk budgeting optimization
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
    return J

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

#Set Dates and Variables for Quarterly Rebalance and Risk Lookback window
f_date = datetime.date(2000, 12, 31) #start date
l_date = datetime.date(2019, 12, 31) #end date
delta = l_date - f_date
quarters_delta = np.floor(delta.days/(365/4)) #number of quarters
quarters_delta = int(quarters_delta) #convert to integer
first_quarter_trading = str('2001-06-30') #chose the first date to start trading

#initialize dataframe

results = pd.DataFrame()
Portfolio_Index_Combo = pd.DataFrame()
 
Data_for_Portfolio_master_prices = PRICES
Data_for_Portfolio_master_returns = PRICES.pct_change().dropna()

for LOOKBACK in range(10,100,10):
        
    portfolio_returns = pd.DataFrame()
    for i in range(0,quarters_delta,1):    
        
        
        #Trading Days
        Start_Date = pd.to_datetime(first_quarter_trading) + pd.tseries.offsets.QuarterEnd(i)
        Start_Date_Trading = Start_Date.strftime('%Y-%m-%d')
        End_Date = pd.to_datetime(Start_Date) + pd.tseries.offsets.QuarterEnd(1)
        End_Date_Trading = End_Date.strftime('%Y-%m-%d')
    
        Start_Date_Trading
        End_Date_Trading
        
        #Filter Days to Use for lookback calculating return and variance
        Risk_Start_Date = pd.to_datetime(Start_Date) + pd.DateOffset(days=-LOOKBACK)
        Risk_Start_Date_1 = Risk_Start_Date.strftime('%Y-%m-%d')
        Risk_End_Date = pd.to_datetime(Start_Date) + pd.DateOffset(days=-1)
        #using a 3 month lookback window to measure risk/returns for optimization
        Risk_End_Date_1 = Risk_End_Date.strftime('%Y-%m-%d')
    
        Risk_Start_Date_1
        Risk_End_Date_1
    
        #filter out quarter only data
        Data_for_Portfolio_master_prices_filter = pd.DataFrame(Data_for_Portfolio_master_prices[Risk_Start_Date_1:Risk_End_Date_1])
        Data_for_Portfolio_master_returns_filter = pd.DataFrame(Data_for_Portfolio_master_returns[Start_Date_Trading:End_Date_Trading])

        Data_for_Portfolio_master_prices_filter.head()
        Data_for_Portfolio_master_prices_filter.tail()
        Data_for_Portfolio_master_returns_filter.head()
        Data_for_Portfolio_master_returns_filter.tail()
        # Calculate expected returns and sample covariance using previous quarter data
        column_length = Data_for_Portfolio_master_prices_filter.shape[1] #number of assets

        #create array of prices and then covariance matrix
        list = []
        for i in range(0,column_length,1):
            prices_array = pd.array(Data_for_Portfolio_master_prices_filter.iloc[:,i]) 
            list.append(prices_array)
        price_array = np.asarray(list)
       
        covariance = np.cov(price_array)

        weights = np.repeat(1/column_length, column_length)
        risk_budget = np.repeat(1/column_length, column_length)
        # your risk budget percent of total portfolio risk (equal risk)
        ERC_weights = np.repeat(1/column_length, column_length) #equal risk for number of assets
        cons = ({'type': 'eq', 'fun': total_weight_constraint},
                {'type': 'ineq', 'fun': long_only_constraint})
    
        V = covariance
        res= minimize(risk_budget_objective, risk_budget, args=[V,ERC_weights], method='SLSQP',constraints=cons, options={'disp': True})
    
        w_rb = np.asmatrix(res.x)
        w_rb #weights for ERC portfolio
    
        #convert to array and then reshape so 11 rows and 1 column to list
        portfolio_weights = np.array(w_rb).reshape((-1, 1)).tolist()
        
        #use dot product to calculate portfolio returns
        return_series = Data_for_Portfolio_master_returns_filter.dot(portfolio_weights)
        returns_df = pd.DataFrame(return_series)
        
        returns_df.head()
        returns_df.tail()
        returns_df.shape
        
        portfolio_returns = portfolio_returns.append(returns_df)
    
        
    Days = len(portfolio_returns)
    perf = portfolio_returns.add(1).prod() ** (252 / Days) - 1 
    std = portfolio_returns.std() * 252 ** 0.5
    Sharpe = perf/std
    portfolio_index = (1 + portfolio_returns).cumprod()
    portfolio_index.shape
    portfolio_index.columns = ['Lookback ' + str(LOOKBACK)]
    results = results.append(pd.DataFrame(
                {'Lookback': LOOKBACK,
                 'STRATEGY': perf,
                 'SHARPE': Sharpe},
                 index = [0]), ignore_index = True)
    Portfolio_Index_Combo = pd.concat([Portfolio_Index_Combo, pd.DataFrame(portfolio_index)], axis = 1, ignore_index = False)


results  #table of results

#plot of the different portfolio lookback periods
Portfolio_Index_Combo.plot()
plt.plot(Portfolio_Index_Combo)
plt.suptitle('Rolling Equal Risk Contribution')
plt.title('Source: Fama French 49 Industry Daily Data', fontsize = 8)
plt.legend(Portfolio_Index_Combo.columns)
plt.show()

#####################SINGLE RUN WITH FF DATA##############################


f_date = datetime.date(1928, 12, 31) #start date
l_date = datetime.date(2019, 12, 31) #end date
delta = l_date - f_date
quarters_delta = np.floor(delta.days/(365/4)) #number of quarters
quarters_delta = int(quarters_delta) #convert to integer
first_quarter_trading = str('1929-06-30') #chose the first date to start trading 
#which needs to be after the Lookback days for risk assessement
LOOKBACK = 100
#initialize dataframe
portfolio_returns = pd.DataFrame()
Data_for_Portfolio_master_prices = PRICES
Data_for_Portfolio_master_returns = PRICES.pct_change().dropna()

for i in range(0,quarters_delta,1):    
             
    #Trading Days
    Start_Date = pd.to_datetime(first_quarter_trading) + pd.tseries.offsets.QuarterEnd(i)
    Start_Date_Trading = Start_Date.strftime('%Y-%m-%d')
    End_Date = pd.to_datetime(Start_Date) + pd.tseries.offsets.QuarterEnd(1)
    End_Date_Trading = End_Date.strftime('%Y-%m-%d')

    Start_Date_Trading
    End_Date_Trading
    
    #Filter Days to Use for lookback calculating return and variance
    Risk_Start_Date = pd.to_datetime(Start_Date) + pd.DateOffset(days=-LOOKBACK)
    Risk_Start_Date_1 = Risk_Start_Date.strftime('%Y-%m-%d')
    Risk_End_Date = pd.to_datetime(Start_Date) + pd.DateOffset(days=-1)
    #using a 3 month lookback window to measure risk/returns for optimization
    Risk_End_Date_1 = Risk_End_Date.strftime('%Y-%m-%d')

    Risk_Start_Date_1
    Risk_End_Date_1

    #filter out quarter only data
    Data_for_Portfolio_master_prices_filter = pd.DataFrame(Data_for_Portfolio_master_prices[Risk_Start_Date_1:Risk_End_Date_1])
    Data_for_Portfolio_master_returns_filter = pd.DataFrame(Data_for_Portfolio_master_returns[Start_Date_Trading:End_Date_Trading])

    Data_for_Portfolio_master_prices_filter.head()
    Data_for_Portfolio_master_prices_filter.tail()
    Data_for_Portfolio_master_returns_filter.head()
    Data_for_Portfolio_master_returns_filter.tail()
    # Calculate expected returns and sample covariance using previous quarter data
    column_length = Data_for_Portfolio_master_prices_filter.shape[1] #number of assets

    #create array of prices and then covariance matrix
    list = []
    for i in range(0,column_length,1):
        prices_array = pd.array(Data_for_Portfolio_master_prices_filter.iloc[:,i]) 
        list.append(prices_array)
    price_array = np.asarray(list)
   
    covariance = np.cov(price_array)

    weights = np.repeat(1/column_length, column_length)
    risk_budget = np.repeat(1/column_length, column_length)
    # your risk budget percent of total portfolio risk (equal risk)
    ERC_weights = np.repeat(1/column_length, column_length) #equal risk for number of assets
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
            {'type': 'ineq', 'fun': long_only_constraint})

    V = covariance
    res= minimize(risk_budget_objective, risk_budget, args=[V,ERC_weights], method='SLSQP',constraints=cons, options={'disp': True})

    w_rb = np.asmatrix(res.x)
    w_rb #weights for ERC portfolio

    #convert to array and then reshape so 11 rows and 1 column to list
    portfolio_weights = np.array(w_rb).reshape((-1, 1)).tolist()
    
    #use dot product to calculate portfolio returns
    return_series = Data_for_Portfolio_master_returns_filter.dot(portfolio_weights)
    returns_df = pd.DataFrame(return_series)
    
    returns_df.head()
    returns_df.tail()
    returns_df.shape
    
    portfolio_returns = portfolio_returns.append(returns_df)
    
portfolio_returns.columns = ['Equal Risk Contribution Portfolio']      
Days = len(portfolio_returns)
perf = portfolio_returns.add(1).prod() ** (252 / Days) - 1 
std = portfolio_returns.std() * 252 ** 0.5
Sharpe = perf/std
Sharpe
perf
portfolio_index = (1 + portfolio_returns).cumprod()
portfolio_index.shape
portfolio_index.plot()

#######################EQUAL WEIGHT#######################################

portfolio_returns_2 = pd.DataFrame() #initialize dataframe
Data_for_Portfolio_master_prices = PRICES
Data_for_Portfolio_master_returns = PRICES.pct_change().dropna()

for i in range(0,quarters_delta,1):    
    
    #Filter Days to Use for calculating return and variance
    Risk_Start_Date = pd.to_datetime(first_quarter_trading) + pd.tseries.offsets.QuarterEnd(i)
    Risk_Start_Date_1 = Risk_Start_Date.strftime('%Y-%m-%d')
    Risk_End_Date = pd.to_datetime(first_quarter_trading) + pd.tseries.offsets.QuarterEnd(i+1)
    #using a 3 month lookback window to measure risk/returns for optimization
    Risk_End_Date_1 = Risk_End_Date.strftime('%Y-%m-%d')
    
    Risk_Start_Date_1
    Risk_End_Date_1
    
    #Trading Days
    Start_Date = pd.to_datetime(Risk_End_Date_1) + pd.DateOffset(days=1)
    Start_Date_Trading = Start_Date.strftime('%Y-%m-%d')
    End_Date = pd.to_datetime(Start_Date) + pd.tseries.offsets.QuarterEnd(1)
    End_Date_Trading = End_Date.strftime('%Y-%m-%d')
    
    Start_Date_Trading
    End_Date_Trading
    
    #filter out quarter only data
    Data_for_Portfolio_master_prices_filter = pd.DataFrame(Data_for_Portfolio_master_prices[Risk_Start_Date_1:Risk_End_Date_1])
    Data_for_Portfolio_master_returns_filter = pd.DataFrame(Data_for_Portfolio_master_returns[Start_Date_Trading:End_Date_Trading])

    Data_for_Portfolio_master_prices_filter.head()
    Data_for_Portfolio_master_prices_filter.tail()
    Data_for_Portfolio_master_returns_filter.head()
    Data_for_Portfolio_master_returns_filter.tail()
    
    Asset_number = Data_for_Portfolio_master_prices_filter.shape[1]  
      
    #equal weight based on number of stocks
    portfolio_weights = np.repeat(1/Asset_number, Asset_number)

    #use dot product to calculate portfolio returns
    return_series = Data_for_Portfolio_master_returns_filter.dot(portfolio_weights)
    returns_df = pd.DataFrame(return_series)
    
    returns_df.head()
    returns_df.tail()
    returns_df.shape
    
    portfolio_returns_2 = portfolio_returns_2.append(returns_df)
            
portfolio_returns_2.columns = ['Equal Weight Portfolio']
Days = len(portfolio_returns_2)
perf = portfolio_returns_2.add(1).prod() ** (252 / Days) - 1 
std = portfolio_returns_2.std() * 252 ** 0.5
Sharpe = perf/std
perf
Sharpe
portfolio_index_2 = (1 + portfolio_returns_2).cumprod()

####COMBINE####

combo_portfolios = pd.concat([portfolio_index, 
                              portfolio_index_2], axis=1, ignore_index=False)
combo_portfolios_nadrop = combo_portfolios.dropna()
combo_portfolios_nadrop.plot()
combo_portfolios_nadrop.tail()
combo_portfolios_nadrop.head()




