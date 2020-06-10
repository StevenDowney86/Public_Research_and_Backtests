#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:42:31 2020

@author: downey
"""

import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web


#use the performance_analysis python file to import functions
os.chdir('PERFORMANCE_ANALYSIS LOCATION HERE')

from performance_analysis import annualized_return
from performance_analysis import annualized_standard_deviation
from performance_analysis import max_drawdown
from performance_analysis import gain_to_pain_ratio
from performance_analysis import calmar_ratio
from performance_analysis import sharpe_ratio
from performance_analysis import sortino_ratio

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)
plt.style.use('ggplot')
#turn off pandas warning for index/slicing as copy warning
pd.options.mode.chained_assignment = None  # default='warn'
#%%
#######################Fundamental and Equity Prices##########################

#fundamental data
fundamental_data = (
    pd.read_csv('file here')
    )

#import all of the equity price data from csv from Sharadar
equity_prices = (
    pd.read_csv('file here')
    )

#get ticker meta data
tickers_df = (
    pd.read_csv('filer here', low_memory=False)
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

#################     Filtering the Dataset        #####################


#test_sector = 'Technology' #

#### The 11 Sectors you can choose from ####

#'Healthcare', 'Basic Materials', 'Financial Services',
#'Technology', 'Industrials', 'Consumer Cyclical', 'Real Estate',
#'Consumer Defensive', 'Communication Services', 'Energy',
#'Utilities'

#If you want to just test the sector
#sector_stocks = tickers_df1[tickers_df1['sector'] == test_sector]

#OR

#If you wanted to remove Real estate and Financial Services
#sector_stocks = tickers_df1[tickers_df1['sector'] != 'Real Estate']
#sector_stocks = sector_stocks[sector_stocks['sector'] != 'Financial Services']
   
#OR
               
#To test all the market
sector_stocks = tickers_df1

#put tickers to list from sector specified
sector_tickers = sector_stocks['ticker'].tolist()

#fundamentals imported already
fundamentals = USA_fundamentals[USA_fundamentals.ticker.isin(sector_tickers)]

#Choose dimension rolling 'twelve month as reported' 'ART'. Sharadar has revisions
#and that would be lookahead bias to use that data.
fundamentals = fundamentals[fundamentals.dimension == 'ART']

#Find data rows where fundamentals have been restated for previous quarter
#and we want to remove for backtesting since at the time you only have the first
#release of data and not subsequent revisions
duplicateRowsDF = fundamentals[fundamentals.duplicated(['ticker', 'calendardate'])]

print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')

fundamentals = fundamentals.drop_duplicates(subset = ['ticker', 'calendardate'],\
                                            keep = 'first')

duplicateRowsDF = fundamentals[fundamentals.duplicated(['ticker', 'calendardate'])]

#make sure there are no duplicates
print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')

#filter out companies with less than $1 billion market cap or another market cap
#that suits your fancy
Data_for_Portfolio = fundamentals[fundamentals['marketcap'] >= 1e9]

#put tickers in a list
tickers = Data_for_Portfolio['ticker'].tolist()
print('There are ' + str(len(set(tickers))) + ' tickers') #number of unique tickers
#%%
#### Map Sector info onto the Fundamental DataFrame to use later ###

#create the dictionary with values and keys as dates
keys = tickers_df1['ticker']
values = tickers_df1['sector']
Dictionary_Sector_values = dict(zip(keys, values))

Data_for_Portfolio['sector'] = Data_for_Portfolio\
    ['ticker'].map(Dictionary_Sector_values)
#%%
#################     Creating Factor Inputs       #####################

### Value Factor ###
Data_for_Portfolio['E/P'] = Data_for_Portfolio['netinc'] / \
    Data_for_Portfolio['marketcap']
Data_for_Portfolio['EBITDA/EV'] = Data_for_Portfolio['ebitda'] / \
    Data_for_Portfolio['ev']
Data_for_Portfolio['FCF/P'] = Data_for_Portfolio['fcf'] / \
    Data_for_Portfolio['marketcap']

### Shareholder Yield ###
Data_for_Portfolio['Shareholder Yield'] = \
    -((Data_for_Portfolio['ncfdebt'] + \
       Data_for_Portfolio['ncfdiv'] + \
           Data_for_Portfolio['ncfcommon']) / Data_for_Portfolio['marketcap'])
    
###### Quality Factor - ideas taken from Alpha Architect QV model #######

####Long Term Business Strength
    
#Can you generate free cash flow
Data_for_Portfolio['FCF/Assets'] = Data_for_Portfolio['fcf'] / \
    Data_for_Portfolio['assets']

#Can you generate returns on investment   
Data_for_Portfolio['ROA'] = Data_for_Portfolio['roa']    
Data_for_Portfolio['ROIC'] = Data_for_Portfolio['roic']

#Do you have a defendable business model?
Data_for_Portfolio['GROSS MARGIN'] = Data_for_Portfolio['grossmargin']

#Current Financial Strength

Data_for_Portfolio['CURRENT RATIO'] = Data_for_Portfolio['currentratio']
Data_for_Portfolio['INTEREST/EBITDA'] = Data_for_Portfolio['intexp'] / \
                                        Data_for_Portfolio['ebitda']    
#%%
###################################################################

t0 = time.time()

#sort out Sector Prices
Sector_stock_prices = equity_prices.loc  \
    [equity_prices['ticker'].isin(tickers)]

Data_for_Portfolio = Data_for_Portfolio.dropna()

#Using the same in sample dates here and for equal weight benchmark
    
f_date = datetime.date(2012, 9, 30) 
l_date = datetime.date(2020, 9, 30) #choosing the last date, results in last
#date for returns is l_date + 1 quarter

delta = l_date - f_date
quarters_delta = np.floor(delta.days/(365/4))
quarters_delta = int(quarters_delta)
first_quarter = str('2012-09-30') #using f_date
Data_for_Portfolio_master = pd.DataFrame(Data_for_Portfolio)

#choose if you want percentiles or fixed number of companies in long portfolio
Percentile_split = .1
#OR
Companies_in_Portfolio = 5
Winsorize_Threshold = .025 #used to determine the winsorize level. If you are
#only going to have a handful of companies than put the threshold really low, 
#otherwise you can use around .025 for a decile portfolio

Portfolio_Turnover = pd.DataFrame()
portfolio_returns = pd.DataFrame()

#extracting and sorting the price index from the stock price df for use
#in the for loop 
price_index = Sector_stock_prices.set_index('date')
price_index = price_index.index
price_index = price_index.unique()
price_index = pd.to_datetime(price_index)
price_index = price_index.sort_values()

for i in range(0, quarters_delta, 4):

    #filter the data for only current date to look at
    Date = pd.to_datetime(first_quarter) + pd.tseries.offsets.QuarterEnd(i)
    Date = Date.strftime('%Y-%m-%d')
    Data_for_Portfolio_master_filter = Data_for_Portfolio_master.loc\
        [Data_for_Portfolio_master['calendardate'] == Date]
    
    ###### VALUE FACTOR ######
    
    #Winsorize the metric data and compress outliers if desired
    Data_for_Portfolio_master_filter['E/P Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['E/P'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['EBITDA/EV Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['EBITDA/EV'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['FCF/P Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['FCF/P'], \
                               limits=Winsorize_Threshold)
    
    #create Z score to normalize the metrics
    Data_for_Portfolio_master_filter['E/P Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['E/P Winsorized'])
    Data_for_Portfolio_master_filter['EBITDA/EV Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['EBITDA/EV Winsorized'])
    Data_for_Portfolio_master_filter['FCF/P Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['FCF/P Winsorized'])
    
    Data_for_Portfolio_master_filter['Valuation Score'] = \
            Data_for_Portfolio_master_filter['E/P Z score'] \
            + Data_for_Portfolio_master_filter['EBITDA/EV Z score']\
            + Data_for_Portfolio_master_filter['FCF/P Z score']
        
    ###### QUALITY FACTOR ######  
    
    Data_for_Portfolio_master_filter['FCF/Assets Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['FCF/Assets'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['ROA Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['ROA'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['ROIC Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['ROIC'], \
                                limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['Gross Margin Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['GROSS MARGIN'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['Current Ratio Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['CURRENT RATIO'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['Interest/EBITDA Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['INTEREST/EBITDA'], \
                               limits=Winsorize_Threshold)
    
    #create Z score
            
    Data_for_Portfolio_master_filter['FCF/Assets Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['FCF/Assets Winsorized'])
    Data_for_Portfolio_master_filter['ROA Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['ROA Winsorized'])
    Data_for_Portfolio_master_filter['ROIC Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['ROIC Winsorized'])
    Data_for_Portfolio_master_filter['Gross Margin Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['Gross Margin Winsorized'])
    Data_for_Portfolio_master_filter['Current Ratio Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['Current Ratio Winsorized'])
    Data_for_Portfolio_master_filter['Interest/EBITDA Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['Interest/EBITDA Winsorized'])
    
    Data_for_Portfolio_master_filter['Quality Score'] = \
        Data_for_Portfolio_master_filter['FCF/Assets Z score'] \
            + Data_for_Portfolio_master_filter['ROA Z score'] \
            + Data_for_Portfolio_master_filter['ROIC Z score']\
            + Data_for_Portfolio_master_filter['Gross Margin Z score']\
            + Data_for_Portfolio_master_filter['Current Ratio Z score']\
            - Data_for_Portfolio_master_filter['Interest/EBITDA Z score']

    ###### SHAREHOLDER YIELD FACTOR #####
    
    Data_for_Portfolio_master_filter['Shareholder Yield Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['Shareholder Yield'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['Shareholder Yield Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['Shareholder Yield Winsorized'])
    Data_for_Portfolio_master_filter['Shareholder Yield Score'] = \
        Data_for_Portfolio_master_filter['Shareholder Yield Z score'] 

    ###### LOW VOLATILITY FACTOR ######
    
    #must have fundamental data from previous factors for price based factors
    #as some equities have price data and no fundamental data which should not
    #be included
    Sector_stocks_Fundamental_tickers = Data_for_Portfolio_master_filter['ticker'].tolist()
  
    Sector_stock_prices_vol_df = Sector_stock_prices.loc\
        [Sector_stock_prices['ticker'].isin(Sector_stocks_Fundamental_tickers)]
   
    Sector_stock_prices_vol_df_1 = Sector_stock_prices_vol_df.iloc[:, [0, 1, 5]]
    
    Sector_stock_prices_vol_df_1_wide = Sector_stock_prices_vol_df_1.pivot\
        (index='date', columns='ticker', values='close')
    
    Sector_stock_prices_vol_df_1_wide = Sector_stock_prices_vol_df_1_wide.fillna(0)
    Sector_stock_returns =  Sector_stock_prices_vol_df_1_wide.pct_change()      
    
    #create rolling vol metric for previous 2 years
    Sector_stock_rolling_vol = Sector_stock_returns.rolling(252*2).std()
    
    #Choose second to last trading day to look at previous vol   
    #Sometimes the dates are off when trying to line up end of quarter and business
    #days so to eliminate errors in the for loop I go to day of quarter, shift forward
    #a business day and then go back two business days
    Date_to_execute_trade = pd.to_datetime(Date) + pd.tseries.offsets.QuarterEnd()
    Date_to_execute_trade_plus1 = Date_to_execute_trade + pd.tseries.offsets.BusinessDay(1)
    final_trade_date = Date_to_execute_trade_plus1 - pd.tseries.offsets.BusinessDay(2)
    
    #pick the final trade date volatility for each ticker
    Filter_Date_Vol = final_trade_date.strftime('%Y-%m-%d')
    Filter_Vol_Signal = Sector_stock_rolling_vol.loc[Filter_Date_Vol]
    Filter_Vol_Signal_Sort = Filter_Vol_Signal.sort_values().dropna()
    
    #create z score and rank for the Volatility Factor
    frame = { 'Vol': Filter_Vol_Signal_Sort} 
    Filter_Vol_Signal_df = pd.DataFrame(frame)
    Filter_Vol_Signal_df['Vol Z Score'] = stats.zscore(Filter_Vol_Signal_Sort)
    Filter_Vol_Signal_df = Filter_Vol_Signal_df.reset_index()
       
    Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.merge(Filter_Vol_Signal_df, how = 'inner', on = ['ticker']) 

    ###### TREND FACTOR #####
    
    tickers_trend = list(Sector_stock_prices_vol_df_1_wide.columns)
    
    #This is a very simply way to see how much a stock is in a trend up or down
    #You could easily make this more complex/robust but it would cost you in 
    #execution time
    df_sma_50 = Sector_stock_prices_vol_df_1_wide.rolling(50).mean()
    df_sma_100 = Sector_stock_prices_vol_df_1_wide.rolling(100).mean()
    df_sma_150 = Sector_stock_prices_vol_df_1_wide.rolling(150).mean()
    df_sma_200 = Sector_stock_prices_vol_df_1_wide.rolling(200).mean()
    
    #Get the same date for vol measurement near rebalance date
    Filter_Date_Trend = final_trade_date.strftime('%Y-%m-%d')
    Filter_Trend_Signal_50 = df_sma_50.loc[Filter_Date_Trend]
    Filter_Trend_Signal_100 = df_sma_100.loc[Filter_Date_Trend]
    Filter_Trend_Signal_150 = df_sma_150.loc[Filter_Date_Trend]
    Filter_Trend_Signal_200 = df_sma_200.loc[Filter_Date_Trend]
    
    Price_Signal = Sector_stock_prices_vol_df_1_wide.loc[Filter_Date_Trend]
    
    Filter_SMA_Signal_df = pd.DataFrame(tickers_trend)
    Filter_SMA_Signal_df = Filter_SMA_Signal_df.rename(columns={0: "ticker"})
    Filter_SMA_Signal_df['SMA 50 position'] = np.where(Price_Signal > Filter_Trend_Signal_50,1,0)
    Filter_SMA_Signal_df['SMA 100 position'] = np.where(Price_Signal > Filter_Trend_Signal_100,1,0)
    Filter_SMA_Signal_df['SMA 150 position'] = np.where(Price_Signal > Filter_Trend_Signal_150,1,0)
    Filter_SMA_Signal_df['SMA 200 position'] = np.where(Price_Signal > Filter_Trend_Signal_200,1,0)
    Filter_SMA_Signal_df['Trend Score'] = np.mean(Filter_SMA_Signal_df, axis=1)
    Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.merge(Filter_SMA_Signal_df[['ticker','Trend Score']], how = 'inner', on = ['ticker'])
    
    ###### MOMENTUM FACTOR #####
    
    tickers_momentum = list(Sector_stock_prices_vol_df_1_wide.columns)
    #from the academic literature of 12 months - 1 month momentum 
    df_mom_11_months = Sector_stock_prices_vol_df_1_wide.pct_change(22*11)
    
    Filter_Date_Mom = Date_to_execute_trade_plus1 - pd.tseries.offsets.BusinessDay(24)
    Filter_Date_Mom_trim = final_trade_date.strftime('%Y-%m-%d')
    Filter_Mom_Signal = df_mom_11_months.loc[Filter_Date_Mom_trim]
     
    Filter_MOM_df = pd.DataFrame(tickers_momentum)
    Filter_MOM_df = Filter_MOM_df.rename(columns={0: "ticker"})
    Filter_MOM_df['Percent Change'] = Filter_Mom_Signal.values
    
    Filter_MOM_df = Filter_MOM_df.replace([np.inf, -np.inf], np.nan)
    Filter_MOM_df = Filter_MOM_df.dropna()
    Filter_MOM_df['Momentum Score'] = stats.zscore(Filter_MOM_df['Percent Change'])
  
    Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.merge(Filter_MOM_df[['ticker','Momentum Score']], how = 'inner', on = ['ticker'])
    
    ### Create Composite Score from factors ###
    
    #Because we made all the factors with a z score each factor should have equal
    #weight in the composite. You could consider changing the weights based on 
    #historical statistical significance or whatever else seems reasonable
    
    #This particular scoring system only invests in companies with 
    #positive trend/momentum after ranking by the other factors
    Data_for_Portfolio_master_filter['Total Score'] = \
        Data_for_Portfolio_master_filter['Valuation Score'] +  \
        Data_for_Portfolio_master_filter['Quality Score'] + \
        Data_for_Portfolio_master_filter['Shareholder Yield Score'] -  \
        Data_for_Portfolio_master_filter['Vol Z Score'] * \
        (Data_for_Portfolio_master_filter['Momentum Score'] + \
        Data_for_Portfolio_master_filter['Trend Score'])
    
    number_firms = Data_for_Portfolio_master_filter.shape
    number_firms = number_firms[0]
    
    firms_in_percentile = np.round(Percentile_split * number_firms)
    
    Data_for_Portfolio_master_filter = \
        Data_for_Portfolio_master_filter.sort_values('Total Score', ascending=False)
    
    ##### How to Filter Companies ####
        
    # Filter so pick the best and worst company from each sector #    
    filtered_df = Data_for_Portfolio_master_filter.copy()
    filtered_df2 = Data_for_Portfolio_master_filter.copy()

    Sector_stocks_cheapest = \
        filtered_df.drop_duplicates(['sector'], keep='first').groupby('ticker').head()
    Sector_stocks_expensive = \
        filtered_df2.drop_duplicates(['sector'], keep='last').groupby('ticker').head()
        
    ######  OR   #######    
    
    #filter the dataset by the desired number of companies for expensive and cheap
    # Sector_stocks_cheapest = Data_for_Portfolio_master_filter.iloc[:int(Companies_in_Portfolio)]
    # Sector_stocks_expensive = Data_for_Portfolio_master_filter.iloc[int(-Companies_in_Portfolio):]
    
    ######  OR   #######
    
    #filter the dataset by the percentile for expensive and cheap
    #Sector_stocks_cheapest = Data_for_Portfolio_master_filter.iloc[:int(firms_in_percentile)]
    #### If you want to create top half portfolio and bottom half ###
    #left_over_firms = number_firms - firms_in_percentile
    #Sector_stocks_expensive = Data_for_Portfolio_master_filter.iloc[int(-left_over_firms):]
    #Sector_stocks_expensive = Data_for_Portfolio_master_filter.iloc[int(-firms_in_percentile):]
       
    #convert the list of unique tickers to a list
    Sector_stocks_cheapest_tickers = Sector_stocks_cheapest['ticker'].tolist()
    Sector_stocks_expensive_tickers = Sector_stocks_expensive['ticker'].tolist()
    
    #keep track of stocks, Tranche, and turnover
    Turnover = pd.DataFrame({'Date':Date,
                             'Tickers':Sector_stocks_cheapest_tickers,
                             'Sector':Sector_stocks_cheapest['sector'].tolist(),
                             'Weight':1/len(Sector_stocks_cheapest_tickers)})
    Portfolio_Turnover = Portfolio_Turnover.append(Turnover)
    
    #filter the price date by the list of tickers
    Sector_stock_prices_cheapest = Sector_stock_prices.loc\
        [Sector_stock_prices['ticker'].isin(Sector_stocks_cheapest_tickers)]
    Sector_stock_prices_expensive = Sector_stock_prices.loc\
        [Sector_stock_prices['ticker'].isin(Sector_stocks_expensive_tickers)]
    
    #get date, ticker, and close(adjusted) columns
    Sector_stock_prices_cheapest = Sector_stock_prices_cheapest.iloc[:, [0, 1, 5]]
    Sector_stock_prices_expensive = Sector_stock_prices_expensive.iloc[:, [0, 1, 5]]
    
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
    Sector_stock_prices_cheapest_wide = Sector_stock_prices_cheapest.pivot\
        (index='date', columns='ticker', values='close')
    Sector_stock_prices_cheapest_wide = \
        Sector_stock_prices_cheapest_wide.fillna(0)
    Sector_stock_prices_expensive_wide = \
        Sector_stock_prices_expensive.pivot(index='date', columns='ticker', values='close')
    Sector_stock_prices_expensive_wide = \
        Sector_stock_prices_expensive_wide.fillna(0)
    
    #This is how to handle if there are no firms in the cheap or empty dataframe.
    #As it stands now, this won't happen, but when in early development
    #There were times when no companies meant the fundamental filter
    if Sector_stock_prices_cheapest_wide.empty == True:
        days = price_index
        number_days = len(price_index)
        filler_returns = np.repeat(0, number_days)
        df = pd.DataFrame({'Days': days, 'Returns': filler_returns})
        df = df.set_index('Days')
        Sector_stock_prices_cheapest_wide = df
    else:
        pass
    
    if Sector_stock_prices_expensive_wide.empty == True:
        days = price_index
        number_days = len(price_index)
        filler_returns = np.repeat(0, number_days)
        df = pd.DataFrame({'Days': days, 'Returns': filler_returns})
        df = df.set_index('Days')
        Sector_stock_prices_expensive_wide = df
    else:
        pass
    
    ###########      High Factor Price Returns and Portfolio     #########
    
    #pick out start date and end date for calculating equity returns
    Sector_stock_prices_cheapest_wide.loc[start_date:end_date].head()
    Sector_stock_prices_cheapest_wide = \
        Sector_stock_prices_cheapest_wide.loc[start_date:end_date]
    Cheap_returns_daily = Sector_stock_prices_cheapest_wide.pct_change()
       
    #if there are no assets then the Cheap_returns_daily become NaN and in that
    #case we will need to by pass the normal operations and just keep the dataframe
    x = np.logical_and\
        (Cheap_returns_daily.shape[1] == 1, Cheap_returns_daily.isnull().all() == True)
    
    if x[0] == True:
        Cheap_returns_daily = Sector_stock_prices_cheapest_wide
    else:
        #get rid of first NaN row
        Cheap_returns_daily = Cheap_returns_daily.dropna(how='all')
    
        #get rid of stocks that have no trading
        Cheap_returns_daily = Cheap_returns_daily.dropna(axis='columns')
    
    column_length = Cheap_returns_daily.shape[1] #if no column
    
    ###### Portfolio Securities Weighting / Optimization #####
    
    #This is where you could add your own portfolio optimization
    
    ##### Equal Weight ####
    
    #equal weight based on number of stocks
    portfolio_weights = np.repeat(1/column_length, column_length)
    
    ### OR ###
    
    #### Market Cap Weight ####
    # Cheap_returns_tickers = Cheap_returns_daily.columns.tolist()
    # Cheap_returns_tickers_fundamentals = Sector_stocks_cheapest.loc[Sector_stocks_cheapest['ticker'].isin(Cheap_returns_tickers)]
    # Cheap_returns_tickers_fundamentals['weights']= Cheap_returns_tickers_fundamentals['marketcap'] / Cheap_returns_tickers_fundamentals['marketcap'].sum()
    # portfolio_weights = np.array(Cheap_returns_tickers_fundamentals['weights'])
    
    ### OR ###
    
    ### Score Confidence Weight ###
    # Cheap_returns_tickers = Cheap_returns_daily.columns.tolist()
    # Cheap_returns_tickers_fundamentals = Sector_stocks_cheapest.loc[Sector_stocks_cheapest['ticker'].isin(Cheap_returns_tickers)]
    # Cheap_returns_tickers_fundamentals['weights']= Cheap_returns_tickers_fundamentals['Total Score'] / Cheap_returns_tickers_fundamentals['Total Score'].sum()
    # portfolio_weights = np.array(Cheap_returns_tickers_fundamentals['weights'])
    
    #use dot product to calculate portfolio returns
    Cheap_returns_daily['portfolio return'] = Cheap_returns_daily.dot(portfolio_weights)
    Portfolio_returns_Cheap = Cheap_returns_daily['portfolio return']
    Portfolio_returns_Cheap = pd.DataFrame(Portfolio_returns_Cheap)
    
    #######       Expensive Stock Price Returns and Portfolio  ########
    
    Sector_stock_prices_expensive_wide = \
        Sector_stock_prices_expensive_wide.loc[start_date:end_date]
    Expensive_returns_daily = Sector_stock_prices_expensive_wide.pct_change()
    
    #if there are no assets then the Cheap_returns_daily become NaN and in that
    #case we will need to by pass the normal operations and just keep the dataframe
    x = np.logical_and(Expensive_returns_daily.shape[1] == 1, \
                        Expensive_returns_daily.isnull().all() == True)
    
    if x[0] == True:
        Expensive_returns_daily = Sector_stock_prices_expensive_wide
    else:
        #get rid of first NaN row
        Expensive_returns_daily = Expensive_returns_daily.dropna(how='all')
    
        #get rid of stocks that have no trading
        Expensive_returns_daily = Expensive_returns_daily.dropna(axis='columns')
    
    column_length = Expensive_returns_daily.shape[1]
    #here you could replace with different ocptimization techniques
    
    #equal weight based on number of stocks
    portfolio_weights = np.repeat(1/column_length, column_length)
    
    ### OR ###
    
    #### Market Cap Weight ####
    # Cheap_returns_tickers = Cheap_returns_daily.columns.tolist()
    # Cheap_returns_tickers_fundamentals = Sector_stocks_cheapest.loc[Sector_stocks_cheapest['ticker'].isin(Cheap_returns_tickers)]
    # Cheap_returns_tickers_fundamentals['weights']= Cheap_returns_tickers_fundamentals['marketcap'] / Cheap_returns_tickers_fundamentals['marketcap'].sum()
    # portfolio_weights = np.array(Cheap_returns_tickers_fundamentals['weights'])
    
    ### OR ###
    
    ### Score Confidence Weight ###
    # Expensive_returns_tickers = Expensive_returns_daily.columns.tolist()
    # Expensive_returns_tickers_fundamentals = Sector_stocks_expensive.loc[Sector_stocks_expensive['ticker'].isin(Expensive_returns_tickers)]
    # Expensive_returns_tickers_fundamentals['weights']= -Expensive_returns_tickers_fundamentals['Total Score'] / Expensive_returns_tickers_fundamentals['Total Score'].sum()
    # portfolio_weights = np.array(Expensive_returns_tickers_fundamentals['weights'])
       
    #use dot product to calculate portfolio returns
    Expensive_returns_daily['portfolio return'] = Expensive_returns_daily.dot(portfolio_weights)
    Portfolio_returns_Expensive = Expensive_returns_daily['portfolio return']
    Portfolio_returns_Expensive = pd.DataFrame(Portfolio_returns_Expensive)
    
    merged = pd.merge(Portfolio_returns_Cheap, Portfolio_returns_Expensive, \
                    how='inner', left_index=True, right_index=True)
    merged['L/S'] = merged.iloc[:, 0]-merged.iloc[:, 1]
    portfolio_returns = portfolio_returns.append(merged)

portfolio_returns.columns = ['High Factor', 'Low Factor', 'LS']

###############   Create the Equal Weight Portfolio Benchmark  ###############
#Equal Weight Portfolio as Benchmark

f_date = datetime.date(2012, 9, 30)
l_date = datetime.date(2020, 9, 30)
delta = l_date - f_date
quarters_delta = np.floor(delta.days/(365/4))
quarters_delta = int(quarters_delta)
first_quarter = str('2012-09-30')
equal_weight_returns = pd.DataFrame()
Data_for_Portfolio_master = pd.DataFrame(Data_for_Portfolio)

for i in range(0, quarters_delta, 1):

    Date = pd.to_datetime(first_quarter) + pd.tseries.offsets.QuarterEnd(i)
    Date = Date.strftime('%Y-%m-%d')
    Data_for_Portfolio_master_filter = \
        Data_for_Portfolio_master.loc[Data_for_Portfolio_master\
                                      ['calendardate'] == Date]
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
    #get rid of first NaN row
    returns_daily = returns_daily.dropna(how='all')
    #get rid of if no column
    returns_daily = returns_daily.dropna(axis='columns')
    column_length = np.where(returns_daily.shape[1] == 0, 1, returns_daily.shape[1])

    portfolio_weights = np.repeat(1/column_length, column_length)
    returns_daily['portfolio return'] = returns_daily.dot(portfolio_weights)
    EW_returns = returns_daily['portfolio return']
    EW_returns_df = pd.DataFrame(EW_returns)
    equal_weight_returns = equal_weight_returns.append(EW_returns_df)

#change inf to NAs since first entry is inf
equal_weight_returns = equal_weight_returns.replace([np.inf, -np.inf], 0)

Combined_Portfolio_returns = pd.merge(equal_weight_returns, \
                                      portfolio_returns, how='inner', \
                                          left_index=True, right_index=True)

#change inf to NAs since first entry is inf
Combined_Portfolio_returns = Combined_Portfolio_returns.replace\
    ([np.inf, -np.inf], 0)

#Rename the columns
Combined_Portfolio_returns = Combined_Portfolio_returns.rename\
    (columns={"portfolio return": "Equal Weight"})

Combined_Portfolio_returns.index = pd.to_datetime(Combined_Portfolio_returns.index)

#create a performance chart and save for later
portfolio_index = (1 + Combined_Portfolio_returns).cumprod()
ax = portfolio_index.plot(title='Multi-Factor Out of Sample performance')
fig = ax.get_figure()
Crystal_Ball_Performance_Chart = 'Multi-Factor Out of Sample Performance Chart Annual Rebal'
path_to_file = (
    r'file here'
        )
output_name = path_to_file + Crystal_Ball_Performance_Chart + '.pdf'
fig.savefig(output_name)

t1 = time.time()
total= t1-t0
print("It took " + str(np.round(total/60,2)) + " minutes to run the code")
#%%
####################get risk free rate from kenneth french#####################
len(get_available_datasets())

ds = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1990-08-30')

print(ds['DESCR'])

ds[0].head()
#%%
data = ds[0]
data = data.dropna()
data = data/100 #convert to percent returns
RF_data = (1+data['RF']).cumprod()

RF_start_date = portfolio_index.first_valid_index()
RF_end_date = portfolio_index.last_valid_index()

RF_data = pd.DataFrame(RF_data[RF_start_date:RF_end_date])
#################Calculate Risk and Performance############################
annualized_return(RF_data)
RF_Ann_Return_df = annualized_return(RF_data)
RF_Ann_Return = np.round(float(RF_Ann_Return_df.iloc[:, 1]), 4)

sum(portfolio_returns['LS'])/(portfolio_returns.shape[0]/252)

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
print(Sector_Perf)
Sector_Perf.set_index('Portfolio')
#%%
#Save the performance for later use
Out_of_Sample_Performance = 'Out of Performance '

path_to_file = (
    r'file_here'
    )
output_name = path_to_file + Out_of_Sample_Performance + '.csv'
Sector_Perf.to_csv(output_name)

os.system('say "your program has finished"')
#%%
####    testing statistical significance of alpha ####

#Sample Size
N = Combined_Portfolio_returns.shape[0]

#Calculate the variance to get the standard deviation
Portfolio_Alpha = Combined_Portfolio_returns['Cheap']- Combined_Portfolio_returns['Equal Weight']
#For unbiased max likelihood estimate we have to divide the var by N-1,
# and therefore the parameter ddof = 1
var_alpha = Portfolio_Alpha.var(ddof=1)

## Calculate the t-statistics
t = (Portfolio_Alpha.mean() - 0) / np.sqrt(var_alpha/N)

## Compare with the critical t-value
#Degrees of freedom
df = N-1

#p-value after comparison with the t
p = 1 - stats.t.cdf(t, df=df)

print("t = " + str(t))
print("p = " + str(p))

cheap_array = np.array(Combined_Portfolio_returns['Cheap'].dropna())

equal_weight_array = np.array(Combined_Portfolio_returns['Equal Weight'])
np.isnan(cheap_array).any()
np.isnan(equal_weight_array).any()
## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(cheap_array, equal_weight_array)
print("t = " + str(t2))
print("p = " + str(p2/2)) #one sided t test
#%%
#####Testing Statistical Significance of L/S Portfolio#########

#Sample Size
N = Combined_Portfolio_returns.shape[0]

#Calculate the variance to get the standard deviation
#For unbiased max likelihood estimate we have to divide the var by N-1,
#and therefore the parameter ddof = 1
var_factor = Combined_Portfolio_returns['LS'].var(ddof=1)

## Calculate the t-statistics
t = (Combined_Portfolio_returns['LS'].mean() - 0) / np.sqrt(var_factor/N)

## Compare with the critical t-value
#Degrees of freedom
df = N-1

#p-value after comparison with the t
p = 1 - stats.t.cdf(t, df=df)

print("t = " + str(t))
print("p = " + str(p))
### You can see that after comparing the

ZEROS = [0] * N
## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(Combined_Portfolio_returns['LS'], ZEROS)
print("t = " + str(t2))
print("p = " + str(p2/2)) #one sided t test
#%%

