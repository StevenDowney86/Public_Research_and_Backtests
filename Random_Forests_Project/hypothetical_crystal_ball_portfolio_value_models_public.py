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


#use the performance_analysis python file to import functions
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

##############################################################################

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

#################     Creating Earnings Foresight        #####################
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

#Using the same in sample dates
    
f_date = datetime.date(2000, 6, 30) 

l_date = datetime.date(2012, 6, 30) #choosing the last date, results in last
#date for returns is l_date + 1 quarter

delta = l_date - f_date
quarters_delta = np.floor(delta.days/(365/4))
quarters_delta = int(quarters_delta)
first_quarter = str('2000-06-30') #using f_date
portfolio_tranche_Cheap_returns = pd.DataFrame()
portfolio_tranche_Expensive_returns = pd.DataFrame()
Data_for_Portfolio_master = pd.DataFrame(Data_for_Portfolio)
Percentile_split = .5
Portfolio_Turnover = pd.DataFrame()
Winsorize_Threshold = .05

price_index = Sector_stock_prices.set_index('date')
price_index = price_index.index
price_index = price_index.unique()
price_index = pd.to_datetime(price_index)
price_index = price_index.sort_values()

#I am using the same for loop to create this crystal ball portfolio as the
#Random Forest model for simplicity

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
        Data_for_Portfolio_master_filter['ROIC Z score'] = \
            stats.zscore(Data_for_Portfolio_master_filter['Forward ROIC Winsorized'])
        Data_for_Portfolio_master_filter['E/P Z score'] = \
            stats.zscore(Data_for_Portfolio_master_filter['Forward E/P Winsorized'])
        Data_for_Portfolio_master_filter['EBITDA/EV Z score'] = \
            stats.zscore(Data_for_Portfolio_master_filter['Forward EBITDA/EV Winsorized'])
        Data_for_Portfolio_master_filter['FCF/P Z score'] = \
            stats.zscore(Data_for_Portfolio_master_filter['Forward FCF/P Winsorized'])

        Data_for_Portfolio_master_filter['Valuation Score'] = \
            Data_for_Portfolio_master_filter['ROIC Z score'] \
                + Data_for_Portfolio_master_filter['E/P Z score'] \
                + Data_for_Portfolio_master_filter['EBITDA/EV Z score']\
                + Data_for_Portfolio_master_filter['FCF/P Z score']

        number_firms = Data_for_Portfolio_master_filter.shape
        number_firms = number_firms[0]

        firms_in_percentile = np.round(Percentile_split * number_firms)

        Data_for_Portfolio_master_filter = \
            Data_for_Portfolio_master_filter.sort_values('Valuation Score', ascending=False)

        #filter the dataset by the percentile for expensive and cheap
        Sector_stocks_cheapest = Data_for_Portfolio_master_filter.iloc[:int(firms_in_percentile)]
        left_over_firms = number_firms - firms_in_percentile
        Sector_stocks_expensive = Data_for_Portfolio_master_filter.iloc[int(-left_over_firms):]
        #Sector_stocks_expensive = Data_for_Portfolio_master_filter.iloc[int(-firms_in_percentile):]

        #convert the list of unique tickers to a list
        Sector_stocks_cheapest_tickers = Sector_stocks_cheapest['ticker'].tolist()
        Sector_stocks_expensive_tickers = Sector_stocks_expensive['ticker'].tolist()

        #keep track of stocks and turnover
        Turnover = pd.DataFrame({'Date':Date,
                                 'Tickers':Sector_stocks_cheapest_tickers,
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

        ###########      Cheap Stock Price Returns and Portfolio     #########

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_cheapest_wide.loc[start_date:end_date].head()
        Sector_stock_prices_cheapest_wide = \
            Sector_stock_prices_cheapest_wide.loc[start_date:end_date]
        Sector_stock_prices_cheapest_wide.head()
        Cheap_returns_daily = Sector_stock_prices_cheapest_wide.pct_change()
        Cheap_returns_daily.head()
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

        #equal weight based on number of stocks
        portfolio_weights = np.repeat(1/column_length, column_length)

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

        #use dot product to calculate portfolio returns
        Expensive_returns_daily['portfolio return'] = Expensive_returns_daily.dot(portfolio_weights)
        Portfolio_returns_Expensive = Expensive_returns_daily['portfolio return']
        Portfolio_returns_Expensive = pd.DataFrame(Portfolio_returns_Expensive)
        merged = pd.merge(Portfolio_returns_Cheap, Portfolio_returns_Expensive, \
                        how='inner', left_index=True, right_index=True)
        merged['L/S'] = merged.iloc[:, 0]-merged.iloc[:, 1]
        portfolio_returns = portfolio_returns.append(merged)

    portfolio_returns.columns = [('Cheap Tranche ' + str(Tranche)), \
                                 ('Expensive Tranch ' + str(Tranche)), 'LS']
    portfolio_tranche_Expensive_returns = \
        pd.concat([portfolio_tranche_Expensive_returns, \
                   portfolio_returns.iloc[:,1:2]], axis=1, ignore_index=False)
    portfolio_tranche_Cheap_returns = \
        pd.concat([portfolio_tranche_Cheap_returns, \
                   portfolio_returns.iloc[:,0]], axis=1, ignore_index=False)

#trim the data where we have returns for all tranches
first_date = portfolio_tranche_Cheap_returns.iloc[:, -1].first_valid_index()
last_date = portfolio_tranche_Cheap_returns.iloc[:, 0].last_valid_index()

condensed_Cheap_portfolio = portfolio_tranche_Cheap_returns[first_date:last_date]
condensed_Expensive_portfolio = portfolio_tranche_Expensive_returns[first_date:last_date]

#replace NaN's with zero
condensed_Cheap_portfolio = condensed_Cheap_portfolio.replace(np.nan, 0)
condensed_Expensive_portfolio = condensed_Expensive_portfolio.replace(np.nan, 0)
#Create a Portfolio by averaging the returns from all 4 tranches
condensed_Cheap_portfolio['Combined Cheap Tranche Portfolio'] = \
    condensed_Cheap_portfolio.mean(axis=1)
condensed_Expensive_portfolio['Combined Expensive Tranche Portfolio'] = \
    condensed_Expensive_portfolio.mean(axis=1)

#Combine into one DataFrame
Combo = pd.concat\
    ([condensed_Cheap_portfolio['Combined Cheap Tranche Portfolio'], \
      condensed_Expensive_portfolio['Combined Expensive Tranche Portfolio']], \
     axis=1, ignore_index=False)

###############   Create the Equal Weight Portfolio Benchmark  ###############

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
                                      Combo, how='inner', \
                                          left_index=True, right_index=True)

#change inf to NAs since first entry is inf
Combined_Portfolio_returns = Combined_Portfolio_returns.replace\
    ([np.inf, -np.inf], 0)

#Rename the columns
Combined_Portfolio_returns = Combined_Portfolio_returns.rename\
    (columns={"portfolio return": "Equal Weight"})

Combined_Portfolio_returns['Long / Short Cheap'] = Combined_Portfolio_returns\
    ['Combined Cheap Tranche Portfolio'] - \
        Combined_Portfolio_returns['Combined Expensive Tranche Portfolio']

Combined_Portfolio_returns.index = pd.to_datetime(Combined_Portfolio_returns.index)

#create a performance chart and save for later
portfolio_index = (1 + Combined_Portfolio_returns).cumprod()
ax = portfolio_index.plot(title='Magic Ball In Sample performance')
fig = ax.get_figure()
Crystal_Ball_Performance_Chart = 'Magic Ball In Sample Performance Chart '
path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code, '
        'and Results/Sector Performance/'
        )
output_name = path_to_file + Crystal_Ball_Performance_Chart + '.pdf'
fig.savefig(output_name)
#%%

####################get risk free rate from kenneth french#####################
len(get_available_datasets())

ds = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1990-08-30')

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
RF_data.tail()
#################Calculate Risk and Performance############################
annualized_return(RF_data)
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
Crystal_Ball_Performance = 'Crystal Ball Performance '

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models,'
    ' Code, and Results/Sector Performance/'
    )
output_name = path_to_file + Crystal_Ball_Performance + '.csv'
Sector_Perf.to_csv(output_name)

os.system('say "your program has finished"')
#%%
