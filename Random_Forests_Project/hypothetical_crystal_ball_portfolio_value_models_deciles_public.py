#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:42:31 2020

@author: downey
"""
import os
import datetime

import pandas as pd
import quandl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web

#use the Performance_Analysis python file to import functions
os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code, and Results/')

from performance_analysis import annualized_return
from performance_analysis import annualized_standard_deviation
from performance_analysis import max_drawdown
from performance_analysis import gain_to_pain_ratio
from performance_analysis import calmar_ratio
from performance_analysis import sharpe_ratio
from performance_analysis import sortino_ratio

##############################################################################

#### This file Shows the what would happen if you new all future fundamental
#### data today and could have a crystal ball of the future (unrealistic)
#### and form portfolios in deciles

#############################################################################

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)
plt.style.use('seaborn')
quandl.ApiConfig.api_key = "apikey"
#turn off pandas warning for index/slicing as copy warning
pd.options.mode.chained_assignment = None  # default='warn'
#%%
#######################Fundamental and Equity Prices##########################

#fundamental data
fundamental_data = (
    pd.read_csv('/Users/downey/Dropbox/Holborn Assets'
                '/Providers/Quandl-Sharadar/SHARADAR_SF1_Fundamentals.csv')
    )

#import all of the equity price data from csv from Sharadar
equity_prices = (
    pd.read_csv('/Users/downey/Dropbox/Holborn Assets/Providers'
                '/Quandl-Sharadar/SHARADAR_SEP_Equity_Prices_All.csv')
    )

#get ticker meta data
tickers_df = (
    pd.read_csv('/Users/downey/Dropbox/Holborn Assets/Providers'
                '/Quandl-Sharadar/SHARADAR_TICKERS.csv', low_memory=False)
    )

#filter out companies not based in USA
tickers_df1 = tickers_df[tickers_df['location'].notnull()]
tickers_df1 = tickers_df1[tickers_df1['location'].str.contains("U.S.")]

#select needed columns to filter out sector in fundamental
tickers_df1 = (
    tickers_df1[['ticker', 'sector', 'name',
                 'industry', 'scalemarketcap']]
    )

#create set and list of all tickers
myset_ticker = set(tickers_df1.ticker)
list_tickers = list(myset_ticker)

#filtered USA fundamental data
USA_fundamentals = fundamental_data[fundamental_data['ticker'].isin(list_tickers)]
#%%

#put tickers to list from sector specified
sector_tickers = tickers_df1['ticker'].tolist()

#fundamentals imported already
fundamentals = USA_fundamentals[USA_fundamentals.ticker.isin(sector_tickers)]

#Choose dimension rolling twelve month as reported 'ART'
fundamentals = fundamentals[fundamentals.dimension == 'ART']

#filter out companies with less than $1 billion market cap
Data_for_Portfolio = fundamentals[fundamentals['marketcap'] >= 1*10e8]

#put tickers in a list
tickers = Data_for_Portfolio['ticker'].tolist()
len(tickers)

########creating Earnings Foresight
Data_for_Portfolio['Forward Earnings'] = fundamentals.groupby('ticker')\
    ['netinc'].shift(-4)
Data_for_Portfolio['Forward ROIC'] = fundamentals.groupby('ticker')\
    ['roic'].shift(-4)
Data_for_Portfolio['Forward EBITDA'] = fundamentals.groupby('ticker')\
    ['ebitda'].shift(-4)
Data_for_Portfolio['Forward FCF'] = fundamentals.groupby('ticker')\
    ['fcf'].shift(-4)

Data_for_Portfolio['Forward E/P'] = Data_for_Portfolio['Forward Earnings'] / \
    Data_for_Portfolio['marketcap']
Data_for_Portfolio['Forward EBITDA/EV'] = Data_for_Portfolio['Forward EBITDA'] / \
    Data_for_Portfolio['ev']
Data_for_Portfolio['Forward FCF/P'] = Data_for_Portfolio['Forward FCF'] / \
    Data_for_Portfolio['marketcap']

Data_for_Portfolio = Data_for_Portfolio.dropna()

#sort out Sector Prices
Sector_stock_prices = equity_prices.loc  \
    [equity_prices['ticker'].isin(tickers)]

#testing train and validation data set
f_date = datetime.date(2000, 6, 30) #must have at least 4 quarters after the
#first available fundamental data point. This will equate to the f_date + 4 quarters
#due to the tranche approach + trading 1 quarter after fundamentals available
#and waiting until have data on all four tranches. So with a f_date = 2000-06-30,
#the first date for returns will be 2001-07-03

l_date = datetime.date(2012, 6, 30) #choosing the last date, results in last
#date for returns is l_date + 1 quarter

delta = l_date - f_date
quarters_delta = np.floor(delta.days/(365/4))
quarters_delta = int(quarters_delta)
first_quarter = str('2000-06-30') #using f_date

portfolio_tranche_decile1_returns = pd.DataFrame()
portfolio_tranche_decile2_returns = pd.DataFrame()
portfolio_tranche_decile3_returns = pd.DataFrame()
portfolio_tranche_decile4_returns = pd.DataFrame()
portfolio_tranche_decile5_returns = pd.DataFrame()
portfolio_tranche_decile6_returns = pd.DataFrame()
portfolio_tranche_decile7_returns = pd.DataFrame()
portfolio_tranche_decile8_returns = pd.DataFrame()
portfolio_tranche_decile9_returns = pd.DataFrame()
portfolio_tranche_decile10_returns = pd.DataFrame()

Data_for_Portfolio_master = pd.DataFrame(Data_for_Portfolio)
Percentile_split = .1
Winsorize_Threshold = .00

price_index = Sector_stock_prices.set_index('date')
price_index = price_index.index
price_index = price_index.unique()
price_index = pd.to_datetime(price_index)
price_index = price_index.sort_values()

#create 4 tranches so that each one rebalances 1x a year on different quarters
for Tranche in range(0, 4, 1):

    portfolio_returns = pd.DataFrame()
    first_quarter_adapted = pd.to_datetime(first_quarter) + pd.tseries.offsets.QuarterEnd(Tranche)

    for i in range(0, quarters_delta, 4):

        #filter the data for only current date to look at
        Date = pd.to_datetime(first_quarter_adapted) + pd.tseries.offsets.QuarterEnd(i)
        Date = Date.strftime('%Y-%m-%d')
        Data_for_Portfolio_master_filter = Data_for_Portfolio_master.loc\
            [Data_for_Portfolio_master['calendardate'] == Date]

        #Winsorize the metric data and compress outliers if desired
        Data_for_Portfolio_master_filter['Forward ROIC Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter['Forward ROIC'], \
                                   limits=Winsorize_Threshold)
        Data_for_Portfolio_master_filter['Forward E/P Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter['Forward E/P'], \
                                   limits=Winsorize_Threshold)
        Data_for_Portfolio_master_filter['Forward EBITDA/EV Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter['Forward EBITDA/EV'], \
                                   limits=Winsorize_Threshold)
        Data_for_Portfolio_master_filter['Forward FCF/P Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter['Forward FCF/P'], \
                                   limits=Winsorize_Threshold)

        #create Z score
        Data_for_Portfolio_master_filter['ROIC Z score'] = stats.zscore\
            (Data_for_Portfolio_master_filter['Forward ROIC Winsorized'])
        Data_for_Portfolio_master_filter['E/P Z score'] = stats.zscore\
            (Data_for_Portfolio_master_filter['Forward E/P Winsorized'])
        Data_for_Portfolio_master_filter['EBITDA/EV Z score'] = stats.zscore\
            (Data_for_Portfolio_master_filter['Forward EBITDA/EV Winsorized'])
        Data_for_Portfolio_master_filter['FCF/P Z score'] = stats.zscore\
            (Data_for_Portfolio_master_filter['Forward FCF/P Winsorized'])

        Data_for_Portfolio_master_filter['Valuation Score'] = \
            Data_for_Portfolio_master_filter['ROIC Z score'] \
                + Data_for_Portfolio_master_filter['E/P Z score'] \
                + Data_for_Portfolio_master_filter['EBITDA/EV Z score']\
                + Data_for_Portfolio_master_filter['FCF/P Z score']

        number_firms = Data_for_Portfolio_master_filter.shape
        number_firms = number_firms[0]
        firms_in_percentile = np.round(Percentile_split * number_firms)

        Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.sort_values\
            ('Valuation Score', ascending=False)

        #filter the dataset by the percentile for expensive and cheap
        Sector_stocks_decile1 = Data_for_Portfolio_master_filter.iloc\
            [:int(firms_in_percentile)]
        Sector_stocks_decile2 = Data_for_Portfolio_master_filter.iloc\
            [int(firms_in_percentile):2*int(firms_in_percentile)]
        Sector_stocks_decile3 = Data_for_Portfolio_master_filter.iloc\
            [2*int(firms_in_percentile):3*int(firms_in_percentile)]
        Sector_stocks_decile4 = Data_for_Portfolio_master_filter.iloc\
            [3*int(firms_in_percentile):4*int(firms_in_percentile)]
        Sector_stocks_decile5 = Data_for_Portfolio_master_filter.iloc\
            [4*int(firms_in_percentile):5*int(firms_in_percentile)]
        Sector_stocks_decile6 = Data_for_Portfolio_master_filter.iloc\
            [5*int(firms_in_percentile):6*int(firms_in_percentile)]
        Sector_stocks_decile7 = Data_for_Portfolio_master_filter.iloc\
            [6*int(firms_in_percentile):7*int(firms_in_percentile)]
        Sector_stocks_decile8 = Data_for_Portfolio_master_filter.iloc\
            [7*int(firms_in_percentile):8*int(firms_in_percentile)]
        Sector_stocks_decile9 = Data_for_Portfolio_master_filter.iloc\
            [8*int(firms_in_percentile):9*int(firms_in_percentile)]

        #choose for the bottom decile the remaining firms
        left_over_firms = number_firms - (Sector_stocks_decile1.shape[0]+
                                          Sector_stocks_decile2.shape[0]+
                                          Sector_stocks_decile3.shape[0]+
                                          Sector_stocks_decile4.shape[0]+
                                          Sector_stocks_decile5.shape[0]+
                                          Sector_stocks_decile6.shape[0]+
                                          Sector_stocks_decile7.shape[0]+
                                          Sector_stocks_decile8.shape[0]+
                                          Sector_stocks_decile9.shape[0])

        Sector_stocks_decile10 = Data_for_Portfolio_master_filter.iloc[-int(left_over_firms):]
        #Sector_stocks_expensive = Data_for_Portfolio_master_filter.iloc[int(-left_over_firms):]

        #convert the list of unique tickers to a list
        Sector_stocks_decile1_tickers = Sector_stocks_decile1['ticker'].tolist()
        Sector_stocks_decile2_tickers = Sector_stocks_decile2['ticker'].tolist()
        Sector_stocks_decile3_tickers = Sector_stocks_decile3['ticker'].tolist()
        Sector_stocks_decile4_tickers = Sector_stocks_decile4['ticker'].tolist()
        Sector_stocks_decile5_tickers = Sector_stocks_decile5['ticker'].tolist()
        Sector_stocks_decile6_tickers = Sector_stocks_decile6['ticker'].tolist()
        Sector_stocks_decile7_tickers = Sector_stocks_decile7['ticker'].tolist()
        Sector_stocks_decile8_tickers = Sector_stocks_decile8['ticker'].tolist()
        Sector_stocks_decile9_tickers = Sector_stocks_decile9['ticker'].tolist()
        Sector_stocks_decile10_tickers = Sector_stocks_decile10['ticker'].tolist()

        #filter the price date by the list of tickers
        Sector_stock_prices_decile1 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile1_tickers)]
        Sector_stock_prices_decile2 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile2_tickers)]
        Sector_stock_prices_decile3 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile3_tickers)]
        Sector_stock_prices_decile4 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile4_tickers)]
        Sector_stock_prices_decile5 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile5_tickers)]
        Sector_stock_prices_decile6 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile6_tickers)]
        Sector_stock_prices_decile7 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile7_tickers)]
        Sector_stock_prices_decile8 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile8_tickers)]
        Sector_stock_prices_decile9 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile9_tickers)]
        Sector_stock_prices_decile10 = Sector_stock_prices.loc\
            [Sector_stock_prices['ticker'].isin(Sector_stocks_decile10_tickers)]

        #get date, ticker, and close(adjusted) columns
        Sector_stock_prices_decile1 = Sector_stock_prices_decile1.iloc[:, [0, 1, 5]]
        Sector_stock_prices_decile2 = Sector_stock_prices_decile2.iloc[:, [0, 1, 5]]
        Sector_stock_prices_decile3 = Sector_stock_prices_decile3.iloc[:, [0, 1, 5]]
        Sector_stock_prices_decile4 = Sector_stock_prices_decile4.iloc[:, [0, 1, 5]]
        Sector_stock_prices_decile5 = Sector_stock_prices_decile5.iloc[:, [0, 1, 5]]
        Sector_stock_prices_decile6 = Sector_stock_prices_decile6.iloc[:, [0, 1, 5]]
        Sector_stock_prices_decile7 = Sector_stock_prices_decile7.iloc[:, [0, 1, 5]]
        Sector_stock_prices_decile8 = Sector_stock_prices_decile8.iloc[:, [0, 1, 5]]
        Sector_stock_prices_decile9 = Sector_stock_prices_decile9.iloc[:, [0, 1, 5]]
        Sector_stock_prices_decile10 = Sector_stock_prices_decile10.iloc[:, [0, 1, 5]]

        #add a quarter to reporting so no lookahead bias
        Date_to_execute_trade = pd.to_datetime(Date) + pd.tseries.offsets.QuarterEnd()
        Date_to_execute_trade_plus1 = Date_to_execute_trade + pd.tseries.offsets.BusinessDay(1)
        final_trade_date = Date_to_execute_trade_plus1 - pd.tseries.offsets.BusinessDay(1)

        #add 4 quarters to end so the rebalance will be annual
        end_date = Date_to_execute_trade + pd.tseries.offsets.QuarterEnd(4)
        final_trade_date_trim = final_trade_date.strftime('%Y-%m-%d')
        end_date_trim = end_date.strftime('%Y-%m-%d')
        start_date = final_trade_date_trim
        end_date = end_date_trim

        #make data from long format to wide and fill in Na's with O
        Sector_stock_prices_decile1_wide = Sector_stock_prices_decile1.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile1_wide = Sector_stock_prices_decile1_wide.fillna(0)

        Sector_stock_prices_decile2_wide = Sector_stock_prices_decile2.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile2_wide = Sector_stock_prices_decile2_wide.fillna(0)

        Sector_stock_prices_decile3_wide = Sector_stock_prices_decile3.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile3_wide = Sector_stock_prices_decile3_wide.fillna(0)

        Sector_stock_prices_decile4_wide = Sector_stock_prices_decile4.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile4_wide = Sector_stock_prices_decile4_wide.fillna(0)

        Sector_stock_prices_decile5_wide = Sector_stock_prices_decile5.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile5_wide = Sector_stock_prices_decile5_wide.fillna(0)

        Sector_stock_prices_decile6_wide = Sector_stock_prices_decile6.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile6_wide = Sector_stock_prices_decile6_wide.fillna(0)

        Sector_stock_prices_decile7_wide = Sector_stock_prices_decile7.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile7_wide = Sector_stock_prices_decile7_wide.fillna(0)

        Sector_stock_prices_decile8_wide = Sector_stock_prices_decile8.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile8_wide = Sector_stock_prices_decile8_wide.fillna(0)

        Sector_stock_prices_decile9_wide = Sector_stock_prices_decile9.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile9_wide = Sector_stock_prices_decile9_wide.fillna(0)

        Sector_stock_prices_decile10_wide = Sector_stock_prices_decile10.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_decile10_wide = Sector_stock_prices_decile10_wide.fillna(0)

        ####    Decile1 Stock Price Returns and Portfolio   ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile1_wide = Sector_stock_prices_decile1_wide.loc[start_date:end_date]

        decile1_returns_daily = Sector_stock_prices_decile1_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile1_returns_daily.shape[1] == 1, \
                           decile1_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile1_returns_daily = Sector_stock_prices_decile1_wide
        else:
            #get rid of first NaN row
            decile1_returns_daily = decile1_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            decile1_returns_daily = decile1_returns_daily.dropna(axis='columns')

        column_length = decile1_returns_daily.shape[1] #if no column

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile1_returns_daily['portfolio return'] = decile1_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile1 = decile1_returns_daily['portfolio return']
        Portfolio_returns_decile1 = pd.DataFrame(Portfolio_returns_decile1)

        ####     Decile2 Stock Price Returns and Portfolio   ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile2_wide = Sector_stock_prices_decile2_wide.loc[start_date:end_date]

        decile2_returns_daily = Sector_stock_prices_decile2_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile2_returns_daily.shape[1] == 1, \
                           decile2_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile2_returns_daily = Sector_stock_prices_decile2_wide
        else:
            #get rid of first NaN row
            decile2_returns_daily = decile2_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            decile2_returns_daily = decile2_returns_daily.dropna(axis='columns')

        column_length = decile2_returns_daily.shape[1] #if no column

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile2_returns_daily['portfolio return'] = decile2_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile2 = decile2_returns_daily['portfolio return']
        Portfolio_returns_decile2 = pd.DataFrame(Portfolio_returns_decile2)

        #####     Decile3 Stock Price Returns and Portfolio    #####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile3_wide = Sector_stock_prices_decile3_wide.loc[start_date:end_date]

        decile3_returns_daily = Sector_stock_prices_decile3_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile3_returns_daily.shape[1] == 1, \
                           decile3_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile3_returns_daily = Sector_stock_prices_decile3_wide
        else:
            #get rid of first NaN row
            decile3_returns_daily = decile3_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            decile3_returns_daily = decile3_returns_daily.dropna(axis='columns')

        column_length = decile3_returns_daily.shape[1] #if no column

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile3_returns_daily['portfolio return'] = decile3_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile3 = decile3_returns_daily['portfolio return']
        Portfolio_returns_decile3 = pd.DataFrame(Portfolio_returns_decile3)

        #####    Decile4 Stock Price Returns and Portfolio    ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile4_wide = Sector_stock_prices_decile4_wide.loc[start_date:end_date]

        decile4_returns_daily = Sector_stock_prices_decile4_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile4_returns_daily.shape[1] == 1, \
                           decile4_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile4_returns_daily = Sector_stock_prices_decile4_wide
        else:
            #get rid of first NaN row
            decile4_returns_daily = decile4_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            decile4_returns_daily = decile4_returns_daily.dropna(axis='columns')

        column_length = decile4_returns_daily.shape[1] #if no column

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile4_returns_daily['portfolio return'] = decile4_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile4 = decile4_returns_daily['portfolio return']
        Portfolio_returns_decile4 = pd.DataFrame(Portfolio_returns_decile4)

        ####    Decile5 Stock Price Returns and Portfolio    ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile5_wide = Sector_stock_prices_decile5_wide.loc[start_date:end_date]

        decile5_returns_daily = Sector_stock_prices_decile5_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile5_returns_daily.shape[1] == 1, \
                           decile5_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile5_returns_daily = Sector_stock_prices_decile5_wide
        else:
            #get rid of first NaN row
            decile5_returns_daily = decile5_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            decile5_returns_daily = decile5_returns_daily.dropna(axis='columns')

        column_length = decile5_returns_daily.shape[1] #if no column

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile5_returns_daily['portfolio return'] = decile5_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile5 = decile5_returns_daily['portfolio return']
        Portfolio_returns_decile5 = pd.DataFrame(Portfolio_returns_decile5)

        ####   Decile6 Stock Price Returns and Portfolio   ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile6_wide = Sector_stock_prices_decile6_wide.loc[start_date:end_date]

        decile6_returns_daily = Sector_stock_prices_decile6_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile6_returns_daily.shape[1] == 1, \
                           decile6_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile6_returns_daily = Sector_stock_prices_decile6_wide
        else:
            #get rid of first NaN row
            decile6_returns_daily = decile6_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            decile6_returns_daily = decile6_returns_daily.dropna(axis='columns')

        column_length = decile6_returns_daily.shape[1] #if no column

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile6_returns_daily['portfolio return'] = decile6_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile6 = decile6_returns_daily['portfolio return']
        Portfolio_returns_decile6 = pd.DataFrame(Portfolio_returns_decile6)

        ####   Decile7 Stock Price Returns and Portfolio   ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile7_wide = Sector_stock_prices_decile7_wide.loc[start_date:end_date]

        decile7_returns_daily = Sector_stock_prices_decile7_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile7_returns_daily.shape[1] == 1, \
                           decile7_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile7_returns_daily = Sector_stock_prices_decile7_wide
        else:
            #get rid of first NaN row
            decile7_returns_daily = decile7_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            decile7_returns_daily = decile7_returns_daily.dropna(axis='columns')

        column_length = decile7_returns_daily.shape[1] #if no column

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile7_returns_daily['portfolio return'] = decile7_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile7 = decile7_returns_daily['portfolio return']
        Portfolio_returns_decile7 = pd.DataFrame(Portfolio_returns_decile7)

        ####   Decile7 Stock Price Returns and Portfolio   ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile8_wide = Sector_stock_prices_decile8_wide.loc[start_date:end_date]

        decile8_returns_daily = Sector_stock_prices_decile8_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile8_returns_daily.shape[1] == 1, \
                           decile8_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile8_returns_daily = Sector_stock_prices_decile8_wide
        else:
            #get rid of first NaN row
            decile8_returns_daily = decile8_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            decile8_returns_daily = decile8_returns_daily.dropna(axis='columns')

        column_length = decile8_returns_daily.shape[1] #if no column
        #here you could replace with different omptimation techniques

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile8_returns_daily['portfolio return'] = decile8_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile8 = decile8_returns_daily['portfolio return']
        Portfolio_returns_decile8 = pd.DataFrame(Portfolio_returns_decile8)

        ####   Decile9 Stock Price Returns and Portfolio   ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile9_wide = Sector_stock_prices_decile9_wide.loc[start_date:end_date]

        decile9_returns_daily = Sector_stock_prices_decile9_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile9_returns_daily.shape[1] == 1, \
                           decile9_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile9_returns_daily = Sector_stock_prices_decile9_wide
        else:
            #get rid of first NaN row
            decile9_returns_daily = decile9_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            decile9_returns_daily = decile9_returns_daily.dropna(axis='columns')

        column_length = decile9_returns_daily.shape[1] #if no column

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile9_returns_daily['portfolio return'] = decile9_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile9 = decile9_returns_daily['portfolio return']
        Portfolio_returns_decile9 = pd.DataFrame(Portfolio_returns_decile9)

        ####   Decile10 Stock Price Returns and Portfolio   ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_decile10_wide = Sector_stock_prices_decile10_wide.loc\
            [start_date:end_date]

        decile10_returns_daily = Sector_stock_prices_decile10_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(decile10_returns_daily.shape[1] == 1, \
                           decile10_returns_daily.isnull().all() == True)

        if x[0] == True:
            decile10_returns_daily = Sector_stock_prices_decile10_wide
        else:
            #get rid of first NaN row
            decile10_returns_daily = decile10_returns_daily.dropna(how='all')
            #get rid of stocks that have no trading
            ecile10_returns_daily = decile10_returns_daily.dropna(axis='columns')

        column_length = decile10_returns_daily.shape[1] #if no column

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

        #use dot product to calculate portfolio returns
        decile10_returns_daily['portfolio return'] = decile10_returns_daily.dot(portfolio_weights)
        Portfolio_returns_decile10 = decile10_returns_daily['portfolio return']
        Portfolio_returns_decile10 = pd.DataFrame(Portfolio_returns_decile10)

        #merge the decile return dataframes
        merged = pd.concat([Portfolio_returns_decile1,
                            Portfolio_returns_decile2,
                            Portfolio_returns_decile3,
                            Portfolio_returns_decile4,
                            Portfolio_returns_decile5,
                            Portfolio_returns_decile6,
                            Portfolio_returns_decile7,
                            Portfolio_returns_decile8,
                            Portfolio_returns_decile9,
                            Portfolio_returns_decile10], axis=1, ignore_index=True)
    portfolio_returns = portfolio_returns.append(merged)

    portfolio_returns.columns = [('Decile 1 Tranche ' + str(Tranche)),
                                 ('Decile 2  Tranch ' + str(Tranche)),
                                 ('Decile 3 Tranche ' + str(Tranche)),
                                 ('Decile 4  Tranch ' + str(Tranche)),
                                 ('Decile 5 Tranche ' + str(Tranche)),
                                 ('Decile 6  Tranch ' + str(Tranche)),
                                 ('Decile 7 Tranche ' + str(Tranche)),
                                 ('Decile 8  Tranch ' + str(Tranche)),
                                 ('Decile 9 Tranche ' + str(Tranche)),
                                 ('Decile 10  Tranch ' + str(Tranche))]
    portfolio_tranche_decile1_returns = \
        pd.concat([portfolio_tranche_decile1_returns, \
                   portfolio_returns.iloc[:, 0]], axis=1, ignore_index=False)
    portfolio_tranche_decile2_returns = \
        pd.concat([portfolio_tranche_decile2_returns, \
                   portfolio_returns.iloc[:, 1:2]], axis=1, ignore_index=False)
    portfolio_tranche_decile3_returns = \
        pd.concat([portfolio_tranche_decile3_returns, \
                   portfolio_returns.iloc[:, 2:3]], axis=1, ignore_index=False)
    portfolio_tranche_decile4_returns = \
        pd.concat([portfolio_tranche_decile4_returns, \
                   portfolio_returns.iloc[:, 3:4]], axis=1, ignore_index=False)
    portfolio_tranche_decile5_returns = \
        pd.concat([portfolio_tranche_decile5_returns, \
                   portfolio_returns.iloc[:, 4:5]], axis=1, ignore_index=False)
    portfolio_tranche_decile6_returns = \
        pd.concat([portfolio_tranche_decile6_returns, \
                   portfolio_returns.iloc[:, 5:6]], axis=1, ignore_index=False)
    portfolio_tranche_decile7_returns = \
        pd.concat([portfolio_tranche_decile7_returns, \
                   portfolio_returns.iloc[:, 6:7]], axis=1, ignore_index=False)
    portfolio_tranche_decile8_returns = \
        pd.concat([portfolio_tranche_decile8_returns, \
                   portfolio_returns.iloc[:, 7:8]], axis=1, ignore_index=False)
    portfolio_tranche_decile9_returns = \
        pd.concat([portfolio_tranche_decile9_returns, \
                   portfolio_returns.iloc[:, 8:9]], axis=1, ignore_index=False)
    portfolio_tranche_decile10_returns = \
        pd.concat([portfolio_tranche_decile10_returns, \
                   portfolio_returns.iloc[:, 9:10]], axis=1, ignore_index=False)

#trim the data where we have returns for all tranches
first_date = portfolio_tranche_decile1_returns.iloc[:, -1].first_valid_index()
last_date = portfolio_tranche_decile1_returns.iloc[:, 0].last_valid_index()

condensed_decile1_portfolio = portfolio_tranche_decile1_returns[first_date:last_date]
condensed_decile2_portfolio = portfolio_tranche_decile2_returns[first_date:last_date]
condensed_decile3_portfolio = portfolio_tranche_decile3_returns[first_date:last_date]
condensed_decile4_portfolio = portfolio_tranche_decile4_returns[first_date:last_date]
condensed_decile5_portfolio = portfolio_tranche_decile5_returns[first_date:last_date]
condensed_decile6_portfolio = portfolio_tranche_decile6_returns[first_date:last_date]
condensed_decile7_portfolio = portfolio_tranche_decile7_returns[first_date:last_date]
condensed_decile8_portfolio = portfolio_tranche_decile8_returns[first_date:last_date]
condensed_decile9_portfolio = portfolio_tranche_decile9_returns[first_date:last_date]
condensed_decile10_portfolio = portfolio_tranche_decile10_returns[first_date:last_date]

#replace NaN's with zero
condensed_decile1_portfolio = condensed_decile1_portfolio.replace(np.nan, 0)
condensed_decile2_portfolio = condensed_decile2_portfolio.replace(np.nan, 0)
condensed_decile3_portfolio = condensed_decile3_portfolio.replace(np.nan, 0)
condensed_decile4_portfolio = condensed_decile4_portfolio.replace(np.nan, 0)
condensed_decile5_portfolio = condensed_decile5_portfolio.replace(np.nan, 0)
condensed_decile6_portfolio = condensed_decile6_portfolio.replace(np.nan, 0)
condensed_decile7_portfolio = condensed_decile7_portfolio.replace(np.nan, 0)
condensed_decile8_portfolio = condensed_decile8_portfolio.replace(np.nan, 0)
condensed_decile9_portfolio = condensed_decile9_portfolio.replace(np.nan, 0)
condensed_decile10_portfolio = condensed_decile10_portfolio.replace(np.nan, 0)

#Create a Portfolio by averaging the returns from all 4 tranches
condensed_decile1_portfolio['Combined Decile1 Tranche Portfolio'] = \
    condensed_decile1_portfolio.mean(axis=1)
condensed_decile2_portfolio['Combined Decile2 Tranche Portfolio'] = \
    condensed_decile2_portfolio.mean(axis=1)
condensed_decile3_portfolio['Combined Decile3 Tranche Portfolio'] = \
    condensed_decile3_portfolio.mean(axis=1)
condensed_decile4_portfolio['Combined Decile4 Tranche Portfolio'] = \
    condensed_decile4_portfolio.mean(axis=1)
condensed_decile5_portfolio['Combined Decile5 Tranche Portfolio'] = \
    condensed_decile5_portfolio.mean(axis=1)
condensed_decile6_portfolio['Combined Decile6 Tranche Portfolio'] = \
    condensed_decile6_portfolio.mean(axis=1)
condensed_decile7_portfolio['Combined Decile7 Tranche Portfolio'] = \
    condensed_decile7_portfolio.mean(axis=1)
condensed_decile8_portfolio['Combined Decile8 Tranche Portfolio'] = \
    condensed_decile8_portfolio.mean(axis=1)
condensed_decile9_portfolio['Combined Decile9 Tranche Portfolio'] = \
    condensed_decile9_portfolio.mean(axis=1)
condensed_decile10_portfolio['Combined Decile10 Tranche Portfolio'] = \
    condensed_decile10_portfolio.mean(axis=1)

#Combine into one DataFrame
Combo = pd.concat([condensed_decile1_portfolio['Combined Decile1 Tranche Portfolio'],
                   condensed_decile2_portfolio['Combined Decile2 Tranche Portfolio'],
                   condensed_decile3_portfolio['Combined Decile3 Tranche Portfolio'],
                   condensed_decile4_portfolio['Combined Decile4 Tranche Portfolio'],
                   condensed_decile5_portfolio['Combined Decile5 Tranche Portfolio'],
                   condensed_decile6_portfolio['Combined Decile6 Tranche Portfolio'],
                   condensed_decile7_portfolio['Combined Decile7 Tranche Portfolio'],
                   condensed_decile8_portfolio['Combined Decile8 Tranche Portfolio'],
                   condensed_decile9_portfolio['Combined Decile9 Tranche Portfolio'],
                   condensed_decile10_portfolio['Combined Decile10 Tranche Portfolio']],
                  axis=1, ignore_index=False)
#%%

###   Create the Equal Weight Portfolio Benchmark ###

#Equal Weight Portfolio as Benchmark
f_date = datetime.date(2000, 6, 30)
l_date = datetime.date(2012, 6, 30)
delta = l_date - f_date
quarters_delta = np.floor(delta.days/(365/4))
quarters_delta = int(quarters_delta)
first_quarter = str('2000-06-30')
equal_weight_returns = pd.DataFrame()
Data_for_Portfolio_master = pd.DataFrame(Data_for_Portfolio)

for i in range(0, quarters_delta, 1):

    Date = pd.to_datetime(first_quarter) + pd.tseries.offsets.QuarterEnd(i)
    Date = Date.strftime('%Y-%m-%d')
    Data_for_Portfolio_master_filter = Data_for_Portfolio_master.loc\
        [Data_for_Portfolio_master['calendardate'] == Date]
    Data_for_Portfolio_master_filter.head()
    Sector_stocks = pd.DataFrame(Data_for_Portfolio_master_filter)
    Sector_stocks_tickers = Sector_stocks['ticker'].tolist()

    Sector_stock_prices_trim = Sector_stock_prices.loc\
        [Sector_stock_prices['ticker'].isin(Sector_stocks_tickers)]
    #get date, ticker, and close(adjusted) columns
    Sector_stock_prices_trim = Sector_stock_prices_trim.iloc[:, [0, 1, 5]]

    #add a quarter to reporting so no lookahead bias
    Date_to_execute_trade = pd.to_datetime(Date) + pd.tseries.offsets.QuarterEnd()
    Date_to_execute_trade_plus1 = Date_to_execute_trade + pd.tseries.offsets.BusinessDay(1)
    final_trade_date = Date_to_execute_trade_plus1 - pd.tseries.offsets.BusinessDay(1)
    #add 4 quarters to end so the rebalance will be annual
    end_date = Date_to_execute_trade + pd.tseries.offsets.QuarterEnd()
    final_trade_date_trim = final_trade_date.strftime('%Y-%m-%d')
    end_date_trim = end_date.strftime('%Y-%m-%d')
    start_date = final_trade_date_trim
    end_date = end_date_trim

    #make data from long format to wide
    Sector_stock_prices_wide = Sector_stock_prices_trim.pivot\
        (index='date', columns='ticker', values='close')
    Sector_stock_prices_wide = Sector_stock_prices_wide.fillna(0)
    #pick out start date and end date for calculating equity returns
    Sector_stock_prices_wide = Sector_stock_prices_wide.loc[start_date:end_date]

    returns_daily = Sector_stock_prices_wide.pct_change()
    returns_daily = returns_daily.dropna(how='all') #get rid of first NaN row
    returns_daily = returns_daily.dropna(axis='columns') #get rid of
    column_length = np.where(returns_daily.shape[1] == 0, 1, returns_daily.shape[1]) #if no column

    portfolio_weights = np.repeat(1/column_length, column_length)
    returns_daily['portfolio return'] = returns_daily.dot(portfolio_weights)
    EW_returns = returns_daily['portfolio return']
    EW_returns_df = pd.DataFrame(EW_returns)
    equal_weight_returns = equal_weight_returns.append(EW_returns_df)

#change inf to NAs since first entry is inf
equal_weight_returns = equal_weight_returns.replace([np.inf, -np.inf], 0)

Combined_Portfolio_returns = \
    pd.merge(equal_weight_returns, Combo, how='inner', left_index=True, right_index=True)

#change inf to NAs since first entry is inf
Combined_Portfolio_returns = Combined_Portfolio_returns.replace([np.inf, -np.inf], 0)

Combined_Portfolio_returns.index = pd.to_datetime(Combined_Portfolio_returns.index)
#create a performance chart and save for later
portfolio_index = (1 + Combined_Portfolio_returns).cumprod()
ax = portfolio_index.plot(title='Magic Ball In Sample performance')
fig = ax.get_figure()
Crystal_Ball_Performance_Chart = 'Magic Ball Decile In Sample Performance Chart '
path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, '
    'Code, and Results/Sector Performance/'
    )
output_name = path_to_file + Crystal_Ball_Performance_Chart + '.pdf'
fig.savefig(output_name)
#%%
####################get risk free rate from kenneth french#####################
len(get_available_datasets())

ds = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1990-08-30',)

print(ds['DESCR'])

ds[0].head()

data = ds[0]
data = data.dropna()
data = data/100 #convert to percent returns
RF_data = (1+data['RF']).cumprod()

RF_start_date = portfolio_index.first_valid_index()
RF_end_date = portfolio_index.last_valid_index()

RF_data = pd.DataFrame(RF_data[RF_start_date:RF_end_date])

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code, and Results/')

#################Calculate Risk and Performance############################
RF_Ann_Return_df = annualized_return(RF_data)
RF_Ann_Return = np.round(float(RF_Ann_Return_df.iloc[:, 1]), 4)

returns = annualized_return(portfolio_index)
Stddev = annualized_standard_deviation(portfolio_index)
Sector_Perf = returns.merge(Stddev)

Sharpe_Ratios = sharpe_ratio(portfolio_index, RF_Ann_Return)
Sector_Perf = Sector_Perf.merge(Sharpe_Ratios)

Sortino_Ratios = sortino_ratio(portfolio_index, RF_Ann_Return)
Sector_Perf = Sector_Perf.merge(Sortino_Ratios)

Max_DD = max_drawdown(portfolio_index)
Sector_Perf = Sector_Perf.merge(Max_DD)

Calmar_Ratios = calmar_ratio(portfolio_index)
Sector_Perf = Sector_Perf.merge(Calmar_Ratios)

Gain_To_Pain = gain_to_pain_ratio(portfolio_index)
Sector_Perf = Sector_Perf.merge(Gain_To_Pain)

#Save the performance for later use

Crystal_Ball_Performance = 'Crystal Ball Decile Performance '

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Sector Performance/'
    )
output_name = path_to_file + Crystal_Ball_Performance + '.csv'
Sector_Perf.to_csv(output_name)

os.system('say "your program has finished"')

print(Sector_Perf)
#%%
