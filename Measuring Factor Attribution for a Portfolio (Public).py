#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:48:51 2019

@author: downey
"""


#This script can be used to see how a funds performance can be attributed
#to the standard factors from Kenneth French and discern if the manager
#generated alpha after factor attribution, as well as if the factors
#are statistically significant


from pandas_datareasder.famafrench import get_available_datasets
import pandas_datareader.data as web
import pandas as pd
from datetime import datetime
from sklearn import linear_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


pd.set_option('display.max_columns',40)

len(get_available_datasets())

#get the portfolio/fund you want to analyze
start=datetime(2010, 1, 1)
end=datetime(2019, 8, 31)

#you will need to get the 'file name' from the the actual url via the French
#website.
ds = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start = start,
                    )
ds_mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start = start,
                    )

print(ds['DESCR'])

ds[0].head()
data = ds[0]
data = data/100

print(ds_mom['DESCR'])

ds_mom[0].head()
data_mom = ds_mom[0]
data_mom = data_mom/100
data_mom = data_mom.rename(columns={'Mom   ':'Mom'})

data_df = pd.DataFrame()
data_df = data.join(data_mom)


#pick the fund, etf, or portfolio you want to measure vs. factors
FUND = web.DataReader("SYLD", "av-daily-adjusted", start, end, access_key = 'Gimmethekey')                                   

fund = pd.DataFrame()
fund['fund returns'] = FUND['adjusted close'].pct_change()

df = pd.DataFrame()
df = data_df.join(fund)
df.dropna(inplace=True)
df.head()
df.columns

X = df[['Mkt-RF','SMB','HML','Mom']] # here we have 3 variables for multiple regression. 
Y = df['fund returns']
 
##################Linear Regression for Factor Attribution####################


#####Stats package and large output#####
X = df[['Mkt-RF','SMB','HML','Mom']] # here we have 3 variables for multiple regression. 
Y = df['fund returns']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.params
results.tvalues
print(results.summary())


########### with sklearn############
X = df[['Mkt-RF','SMB','HML','Mom']] # here we have 3 variables for multiple regression. 
Y = df['fund returns']

regr = linear_model.LinearRegression()
regr.fit(X, Y)
regr.score(X, Y)

print('Mkt-RF, SMB, HML, Mom Beta Coefficient : \n', regr.coef_)
print(regr.intercept_)

unexplained_variation = round(1 - round(regr.score(X, Y),3),3)

#Adjusted R Squared below which is nearly the same
adjusted_R_Squared = round((1 - (1-regr.score(X, Y))*(len(Y)-1)/(len(Y)-X.shape[1]-1)),3)

#amound explained and unexplained by index etfs
print('The factors explain ' + str(adjusted_R_Squared) + ' percent of the portfolio variation' + 
      ' and thus ' + str(unexplained_variation) + ' is unexplained by the factors')

annualized_returns = (1+df).cumprod()**(252/len(df))-1
annualized_returns[-1:]
returns = []
returns = np.array(annualized_returns.iloc[-1:,[0,1,2,4]])
returns


betas = pd.DataFrame()
betas['coeff'] = regr.coef_
betas['perf'] = returns.transpose()
betas['return contribution'] = betas['coeff']*betas['perf']
fund_returns = annualized_returns.iloc[-1,-1]
fund_returns_exess = fund_returns - annualized_returns.iloc[-1,-3]
fund_returns_exess
alpha = fund_returns_exess - betas['return contribution'].sum()

betas.rename({0: 'Mkt-RF', 1: 'SMB', 2: 'HML', 3: 'Mom'})
totals = pd.DataFrame()
totals = betas['return contribution']

alpha_df = pd.DataFrame()
alpha_df = alpha
alpha_df = pd.Series(alpha_df)
totals = totals.append(alpha_df)
fund_returns_excess_df = pd.Series(fund_returns_exess)
totals = totals.append(fund_returns_excess_df)
totals = totals.reset_index(drop=True)
totals = totals.rename({0: 'Mkt-RF', 1: 'SMB', 2: 'HML',3: 'Mom', 4: 'alpha', 5: 'Fund excess Returns'})
totals

#Plots the factor attribution of the portfolio
sns.set()
plt.title("Factor Expsoure for the Fund")
totals.T.plot(kind='bar', stacked=True)

