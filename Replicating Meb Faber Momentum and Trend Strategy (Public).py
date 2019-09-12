#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:16:11 2019

@author: downey
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time
from itertools import product
import seaborn as sns

results = pd.DataFrame()

pd.set_option('display.max_columns', 50)

#parse the dates in monthly fama french
mydateparser = lambda x: pd.datetime.strptime(x, "%Y%m")
raw = pd.read_csv('Coding/Python/Data Files/10_Industry_Portfolios copy.csv', index_col = 0, parse_dates = True, date_parser=mydateparser)

#slice the data to be start - end of 2008 so I can replicate Meb Faber's work
#which ended 2008
raw2 = raw.iloc[0:990,]
raw2.head()
raw2.tail()

raw = raw2

raw.tail()
raw.head(10)

#convert to decimal, which gets to percentage terms
raw = raw/100
raw.describe()

asset_number = raw.shape[1]
asset_number
##########create rolling returns and index values#########
index = (1+raw).cumprod()
index.head()
index.tail()
   
log_returns = np.log(index/index.shift(1))
log_returns.head(10)
log_returns.tail()

#create new column names
nameslist = []
for col in index.columns: 
    nameslist.append(col + str(' log returns'))
    
log_returns.columns = nameslist    

#join dataframes
index = index.join(log_returns, on = 'Date')
column_number_end_returns = index.shape[1]

#choose the number of months look back for momentum
month_lookback = 12
rolling_sum = index.iloc[:,asset_number:].rolling(month_lookback).sum()
rolling_sum.tail(15)


nameslist2 = []
for col in index.iloc[:,0:asset_number].columns: 
    nameslist2.append(col + str(' log ret. roll'))
 
rolling_sum.columns = nameslist2
index = index.join(rolling_sum, on = 'Date')
index.tail()
######TRYING TO IMPLEMENT VIA FOR LOOP##########

#range_values = range(1,1,1)
#top_assets = range(1,11,1)
#names = list(raw.columns.values)

#momentum_df = pd.DataFrame()
#for rangevalues, topassets in product(range_values, top_assets):
    #print('Testing average of length: '+ str(rangevalues))
    #rolling_sum = log_returns.rolling(rangevalues).sum()
    #sortedvalues  = rolling_sum.apply(pd.Series.rank, axis = 1)
    #df3 = rolling_sum
    #df4 = df3.drop(['NoDur','Durbl','Manuf','Enrgy','HiTec','Telcm','Shops','Utils','Hlth ','Other'], axis = 1)
    #Rank_Number = 11 - topassets
    #for string in names:
        #first_column = pd.DataFrame(np.where(sortedvalues[string] >= Rank_Number, 1, 0))
        #df4 = pd.concat([df4, first_column], axis = 1, ignore_index = False)
#df4.head(15)
    
    
    
####################################################
#sort values at each rolling period

sortedvalues  = index.iloc[:,column_number_end_returns:].apply(pd.Series.rank, axis = 1)

sortedvalues.head(15)
sortedvalues.tail(15)

#create columns
nameslist3 = []
for col in index.iloc[:,:asset_number].columns: 
    nameslist3.append(col + str(' rank'))
 
sortedvalues.columns = nameslist3

index = index.join(sortedvalues, on = 'Date')
index.tail()

#the rank is 1 - 10 with 10 being the best, thus 8,9,10 is top 3
#choose the top number of assets out of 10 to include

Top_Assets = 3
#Have to use (asset_number + 1) minus the Top_Assets number in order for the rank to work 
Rank_Number = (asset_number+1) - Top_Assets

df3 = rolling_sum
#couldn't figure out how to create DataFrame with same date index so did it
#this way which isn't pretty
df4 = df3.drop([df3.columns[0],
                df3.columns[1], 
                df3.columns[2],
                df3.columns[3], 
                df3.columns[4],
                df3.columns[5], 
                df3.columns[6],
                df3.columns[7], 
                df3.columns[8],
                df3.columns[9]],axis='columns')
df4.head()

df4['NonDur_allocation'] = np.where(sortedvalues.iloc[:,0] >= Rank_Number, 1, 0)
df4['Durbl_allocation'] = np.where(sortedvalues.iloc[:,1] >= Rank_Number, 1, 0)
df4['Manuf_allocation'] = np.where(sortedvalues.iloc[:,2] >= Rank_Number, 1, 0)
df4['Enrgy_allocation'] = np.where(sortedvalues.iloc[:,3] >= Rank_Number, 1, 0)
df4['HiTec_allocation'] = np.where(sortedvalues.iloc[:,4] >= Rank_Number, 1, 0)
df4['Telcm_allocation'] = np.where(sortedvalues.iloc[:,5] >= Rank_Number, 1, 0)
df4['Shops_allocation'] = np.where(sortedvalues.iloc[:,6] >= Rank_Number, 1, 0)
df4['Utils_allocation'] = np.where(sortedvalues.iloc[:,7] >= Rank_Number, 1, 0)
df4['Hlth_allocation'] = np.where(sortedvalues.iloc[:,8] >= Rank_Number, 1, 0)
df4['Other_allocation'] = np.where(sortedvalues.iloc[:,9] >= Rank_Number, 1, 0)
df4.head(15)
df4.tail()

index = index.join(df4, on = 'Date')
index.tail()
#Create SMA dataframe so that if the chosen assets are in uptrend allocate
#otherwise put in cash

rolling_SMA = df3.drop([df3.columns[0],
                df3.columns[1], 
                df3.columns[2],
                df3.columns[3], 
                df3.columns[4],
                df3.columns[5], 
                df3.columns[6],
                df3.columns[7], 
                df3.columns[8],
                df3.columns[9]],axis='columns')

#easier way to create SMA df
SMA = month_lookback
df_sma = index.iloc[:,0:asset_number].rolling(SMA).mean()
df_sma.tail(15)

#create columns
nameslist4 = []
for col in index.iloc[:,:asset_number].columns: 
    nameslist4.append(col + str(' SMA'))
 
df_sma.columns = nameslist4

index = index.join(df_sma, on = 'Date')
index.tail()

#create new data frame that gives proper asset weight
df5 = df4/Top_Assets
df5.tail()
#create columns
nameslist5 = []
for col in index.iloc[:,:asset_number].columns: 
    nameslist5.append(col + str(' weighted allocation'))
 
df5.columns = nameslist5

index = index.join(df5, on = 'Date')
index.tail()

#invest in assets that are topped rank only if above SMA trend
df7 = df3.drop([df3.columns[0],
                df3.columns[1], 
                df3.columns[2],
                df3.columns[3], 
                df3.columns[4],
                df3.columns[5], 
                df3.columns[6],
                df3.columns[7], 
                df3.columns[8],
                df3.columns[9]],axis='columns')
 
index.columns    
       
df7['NonDur'] = np.where(index.iloc[:,0] >= index.iloc[:,-asset_number*2], index.iloc[:,-asset_number], 0)
df7['Durbl'] = np.where(index.iloc[:,1] >= index.iloc[:,-asset_number*2+1], index.iloc[:,-asset_number+1], 0)
df7['Manuf'] = np.where(index.iloc[:,2] >= index.iloc[:,-asset_number*2+2], index.iloc[:,-asset_number+2], 0)
df7['Enrgy'] = np.where(index.iloc[:,3] >= index.iloc[:,-asset_number*2+3], index.iloc[:,-asset_number+3], 0)
df7['HiTec'] = np.where(index.iloc[:,4] >= index.iloc[:,-asset_number*2+4], index.iloc[:,-asset_number+4], 0)
df7['Telcm'] = np.where(index.iloc[:,5] >= index.iloc[:,-asset_number*2+5], index.iloc[:,-asset_number+5], 0)
df7['Shops'] = np.where(index.iloc[:,6] >= index.iloc[:,-asset_number*2+6], index.iloc[:,-asset_number+6], 0)
df7['Utils'] = np.where(index.iloc[:,7] >= index.iloc[:,-asset_number*2+7], index.iloc[:,-asset_number+7], 0)
df7['Hlth'] = np.where(index.iloc[:,8] >= index.iloc[:,-asset_number*2+8], index.iloc[:,-asset_number+8], 0)
df7['Other'] = np.where(index.iloc[:,9] >= index.iloc[:,-asset_number*2+9], index.iloc[:,-asset_number+9], 0)

#create columns
nameslist6 = []
for col in index.iloc[:,:asset_number].columns: 
    nameslist6.append(col + str(' SMA Filtered Allocation'))
 
df7.columns = nameslist6

index = index.join(df7, on = 'Date')

#multiply the weights in index by the returns in raw returns and shift so no look
#ahead bias. the top ranked this month, those weights apply to next month
#returns

df6 = pd.DataFrame()
df6['NonDur_Port_Return'] = index.iloc[:,-asset_number].shift(1)*raw.iloc[:,0]
df6['Durbl_Port_Return'] = index.iloc[:,-asset_number+1].shift(1)*raw.iloc[:,1]
df6['Manuf_Port_Return'] = index.iloc[:,-asset_number+2].shift(1)*raw.iloc[:,2]
df6['Enrgy_Port_Return'] = index.iloc[:,-asset_number+3].shift(1)*raw.iloc[:,3]
df6['HiTec_Port_Return'] = index.iloc[:,-asset_number+4].shift(1)*raw.iloc[:,4]
df6['Telcm_Port_Return'] = index.iloc[:,-asset_number+5].shift(1)*raw.iloc[:,5]
df6['Shops_Port_Return'] = index.iloc[:,-asset_number+6].shift(1)*raw.iloc[:,6]
df6['Hlth_Port_Return'] = index.iloc[:,-asset_number+7].shift(1)*raw.iloc[:,7]
df6['Util_Port_Return'] = index.iloc[:,-asset_number+8].shift(1)*raw.iloc[:,8]
df6['Other_Port_Return'] = index.iloc[:,-asset_number+9].shift(1)*raw.iloc[:,9]
df6.tail()

#create columns
nameslist7 = []
for col in index.iloc[:,:asset_number].columns: 
    nameslist7.append(col + str(' Str Return'))
 
df6.columns = nameslist7
index = index.join(df6, on = 'Date')

#sum the returns of the portfolio assets
df8 = pd.DataFrame()

df8['Strategy Returns'] = index.iloc[:,-asset_number:].sum(axis = 1, skipna = True)

index = index.join(df8, on = 'Date')

#index of the cumulative geometric returns
performance = (1+index.iloc[:,-1]).cumprod()
perf = performance.iloc[-1,]
print(perf)

#find the number of months in the dataframe
column_length = index.iloc[:,-1].shape[0]
years = column_length/12

#get annualized returns for strategy - getting wrong answer here 
annualized_return = (perf ** (1/years)) - 1
print(annualized_return)

#annualized standard deviation of portfolio
std = index.iloc[:,-1].std() * (12**.5)
print(std)

#repeat the results manually changing assets and lookback window
results = results.append(pd.DataFrame(
                {'Window': month_lookback, "Top Assets": Top_Assets,
                 'Std Dev': std, 
                 'STRATEGY': annualized_return},
                 index = [0]), ignore_index = True)

results

########################################################################

heatmapresults = results
heatmapresults = heatmapresults.pivot("Window","Top Assets","STRATEGY")
heatmapresults.head()
ax = sns.heatmap(heatmapresults, annot = True, cmap="YlGnBu")


######calculate max drawdown#########
# We are going to use a trailing 12 month trading day window
window = 12

# Calculate the max drawdown in the past window days for each day in the series.
# Use min_periods=1 if you want to let the first 252 days data have an expanding window
Roll_Max = pd.DataFrame()
Roll_Max = performance.rolling(window).max()
Monthly_Drawdown = performance/Roll_Max - 1.0

# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window
Max_Monthly_Drawdown = Monthly_Drawdown.rolling(100).min()

# Plot the results
Monthly_Drawdown.plot()
Max_Monthly_Drawdown.plot()

##########################
raw2 = pd.read_csv('/Users/downey/Coding/Python/Python for Finance/F-F_Research_Data_Factors_daily.csv', index_col = 0, parse_dates = True)
raw2.head()
raw2['Market+RF'] = raw2['Mkt-RF'] + raw2['RF']
raw2 = raw2/100
raw2['index'] = (1+raw2['Market+RF']).cumprod()
column_length2 = raw2.shape[0]
last_row = raw2.iloc[-1,-1]
last_row
annualized_return2 = (last_row ** (1/column_length2)) ** (252) - 1
#Annualized Return for the market over similar time frame
print(annualized_return2)
