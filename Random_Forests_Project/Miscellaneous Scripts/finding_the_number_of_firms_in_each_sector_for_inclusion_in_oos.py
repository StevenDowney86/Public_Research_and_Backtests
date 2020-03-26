#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:01:26 2020

@author: downey
"""

import pandas as pd

####    calculating the number of firms on the last quarter per sector   ####

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
tickers_df1 = tickers_df1[['ticker', 'sector', 'name', 'industry', 'scalemarketcap']]

#create set and list of all tickers
myset_ticker = set(tickers_df1.ticker)
list_tickers = list(myset_ticker)

#filtered USA fundamental data
USA_fundamentals = fundamental_data[fundamental_data['ticker'].isin(list_tickers)]

###########################Choosing the Sector to Test#####################

#fitler out the tickers based on specified sector

tickers_df1.sector.unique() #unique sectors
#Healthcare', 'Basic Materials', 'Financial Services','Consumer Cyclical',
#'Technology', 'Industrials', 'Real Estate','Energy', 'Consumer Defensive',
#'Communication Services', 'Utilities'
#%%
test_sector = 'Real Estate' #you will need to cycle through each sector

sector_stocks = tickers_df1[tickers_df1['sector'] == test_sector]

#put tickers to list from sector specified
sector_tickers = sector_stocks['ticker'].tolist()
len(sector_tickers)
#fundamentals imported already
fundamentals = USA_fundamentals[USA_fundamentals.ticker.isin(sector_tickers)]

#filter out companies with less than $1 billion market cap based on previous quarter
fundamentals = fundamentals[fundamentals['marketcap'] >= 1*1e9]

#Choose dimension rolling twelve month as reported 'ART'
fundamentals = fundamentals[fundamentals.dimension == 'ART']

###########################Creating Training/Validation Set###################
fundamentals = fundamentals[fundamentals['calendardate'] == '2012-12-31']

#Find data rows where fundamentals have been restated for previous quarter
#and we want to remove for backtesting since at the time you only have the first
#release of data and not subsequent revisions
duplicateRowsDF = fundamentals[fundamentals.duplicated(['ticker', 'calendardate'])]

#print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')

fundamentals = fundamentals.drop_duplicates\
    (subset=['ticker', 'calendardate'], keep='first')

duplicateRowsDF = fundamentals[fundamentals.duplicated(['ticker', 'calendardate'])]

print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')

#put tickers to list from sector specified
sector_tickers_YE_2012 = fundamentals['ticker'].tolist()
print('There are ' + str(len(sector_tickers_YE_2012)) + ' companies in ' + test_sector)
#%%
