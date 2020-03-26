#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:42:31 2020

@author: downey
"""
import datetime
from scipy import stats
import os
import pandas as pd
import quandl
import numpy as np
import matplotlib.pyplot as plt
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

#### This script combines the 4 value factors from the sector Random Forests
#### models and then chooses the stocks with the best value score based on 
#### that forecast. The resulting portfolio is a combination of the sector 
#### portfolios

##############################################################################


pd.set_option('display.max_columns', 110)
pd.set_option('display.max_rows', 1000)
plt.style.use('seaborn')
quandl.ApiConfig.api_key = "apikey"
#turn off pandas warning for index/slicing as copy warning
pd.options.mode.chained_assignment = None  # default='warn'
#%%
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

tickers_df1.sector.unique() #unique sectors
#Healthcare', 'Basic Materials', 'Financial Services','Consumer Cyclical',
#'Technology', 'Industrials', 'Real Estate','Energy', 'Consumer Defensive',
#'Communication Services', 'Utilities'
#%%
test_sector = 'Healthcare' #you will need to cycle through each sector

#import the forecasted values for the sector

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
         ' and Results/Random Forest Models/ROIC Forecasts/'
         )
Data_for_Portfolio_ROIC = pd.read_csv(str(test_sector) + '_ROIC_Predictions.csv', index_col=[0])

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
         ' and Results/Random Forest Models/PE Forecasts/'
         )
Data_for_Portfolio_PE = pd.read_csv(str(test_sector) + '_PE_Predictions.csv', index_col=[0])

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
         ' and Results/Random Forest Models/EVEBITDA Forecasts/'
         )
Data_for_Portfolio_EVEBITDA = \
    pd.read_csv(str(test_sector) + '_EVEBITDA_Predictions.csv', index_col=[0])

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
         ' and Results/Random Forest Models/PFCF Forecasts/'
         )
Data_for_Portfolio_PFCF = \
    pd.read_csv(str(test_sector) + '_PFCF_Predictions.csv', index_col=[0])

#merge the data
Data_for_Portfolio = Data_for_Portfolio_ROIC.merge(Data_for_Portfolio_PE)
Data_for_Portfolio = Data_for_Portfolio.merge(Data_for_Portfolio_EVEBITDA)
Data_for_Portfolio = Data_for_Portfolio.merge(Data_for_Portfolio_PFCF)

#put tickers in a list
tickers = Data_for_Portfolio['ticker'].tolist()
len(tickers)

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

#create 4 tranches so that each one rebalances 1x a year on different quarters
for Tranche in range(0, 4, 1):

    portfolio_returns = pd.DataFrame()
    first_quarter_adapted = pd.to_datetime(first_quarter) + pd.tseries.offsets.QuarterEnd(Tranche)

    for i in range(0, quarters_delta, 4):

        #fitler the data for only current date to look at
        Date = pd.to_datetime(first_quarter_adapted) + pd.tseries.offsets.QuarterEnd(i)
        Date = Date.strftime('%Y-%m-%d')
        Data_for_Portfolio_master_filter = \
            Data_for_Portfolio_master.loc[Data_for_Portfolio_master['calendardate'] == Date]

        #Winsorize the metric data and compress outliers if desired
        Data_for_Portfolio_master_filter['Forward ROIC Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter\
                                   ['Forward ROIC'], limits=Winsorize_Threshold)
        Data_for_Portfolio_master_filter['Forward E/P Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter\
                                   ['Forward E/P'], limits=Winsorize_Threshold)
        Data_for_Portfolio_master_filter['Forward EBITDA/EV Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter\
                                   ['Forward EBITDA/EV'], limits=Winsorize_Threshold)
        Data_for_Portfolio_master_filter['Forward FCF/P Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter\
                                   ['Forward FCF/P'], limits=Winsorize_Threshold)

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

        left_over_firms = number_firms - firms_in_percentile
        Data_for_Portfolio_master_filter = \
            Data_for_Portfolio_master_filter.sort_values('Valuation Score', ascending=False)

        #filter the dataset by the percentile for expensive and cheap
        Sector_stocks_cheapest = Data_for_Portfolio_master_filter.iloc[:int(firms_in_percentile)]
        Sector_stocks_expensive = Data_for_Portfolio_master_filter.iloc[int(-left_over_firms):]

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
        Sector_stock_prices_cheapest_wide = Sector_stock_prices_cheapest_wide.fillna(0)
        Sector_stock_prices_expensive_wide = Sector_stock_prices_expensive.pivot\
            (index='date', columns='ticker', values='close')
        Sector_stock_prices_expensive_wide = Sector_stock_prices_expensive_wide.fillna(0)

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

        ####     Cheap Stock Price Returns and Portfolio    ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_cheapest_wide.loc[start_date:end_date].head()
        Sector_stock_prices_cheapest_wide = \
            Sector_stock_prices_cheapest_wide.loc[start_date:end_date]
        Sector_stock_prices_cheapest_wide.head()
        Cheap_returns_daily = Sector_stock_prices_cheapest_wide.pct_change()

        #if there are no assets then the Cheap_returns_daily become NaN and in that
        #case we will need to by pass the normal operations and just keep the dataframe
        x = np.logical_and(Cheap_returns_daily.shape[1] == 1, \
                           Cheap_returns_daily.isnull().all() == True)

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

        ####    Expensive Stock Price Returns and Portfolio    ####

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

    portfolio_returns.columns = [('Cheap Tranche ' + str(Tranche)),\
                                 ('Expensive Tranch ' + str(Tranche)), 'LS']

    portfolio_tranche_Expensive_returns = pd.concat\
        ([portfolio_tranche_Expensive_returns, portfolio_returns.iloc[:, 1:2]], \
         axis=1, ignore_index=False)

    portfolio_tranche_Cheap_returns = pd.concat\
        ([portfolio_tranche_Cheap_returns, portfolio_returns.iloc[:, 0]], \
         axis=1, ignore_index=False)

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
    ([condensed_Cheap_portfolio['Combined Cheap Tranche Portfolio'],\
      condensed_Expensive_portfolio['Combined Expensive Tranche Portfolio']],\
     axis=1, ignore_index=False)

####   Create the Equal Weight Portfolio Benchmark    ####

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
    pd.merge(equal_weight_returns, Combo, how='inner', left_index=True, \
             right_index=True)

#change inf to NAs since first entry is inf
Combined_Portfolio_returns = \
    Combined_Portfolio_returns.replace([np.inf, -np.inf], 0)

#Rename the columns
Combined_Portfolio_returns = \
    Combined_Portfolio_returns.rename(columns={"portfolio return": "Equal Weight"})

Combined_Portfolio_returns['Long / Short'] = \
    Combined_Portfolio_returns['Combined Cheap Tranche Portfolio'] - \
        Combined_Portfolio_returns['Combined Expensive Tranche Portfolio']

#Save the Sector Returns to csv for use later
Sector_Returns = 'Sector Returns '+test_sector

path_to_file = (r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models,'
                ' Code, and Results/Sector Performance/'
                )
output_name = path_to_file + Sector_Returns + '.csv'
Combined_Portfolio_returns.to_csv(output_name)

#create a performance chart and save for later
portfolio_index = (1 + Combined_Portfolio_returns).cumprod()
ax = portfolio_index.plot(title='In Sample ' + str(test_sector) + ' performance')
fig = ax.get_figure()
Sector_Performance_Chart = 'In Sample Performance Chart '+test_sector
path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models,'
    ' Code, and Results/Sector Performance/'
                )
output_name = path_to_file + Sector_Performance_Chart + '.pdf'
fig.savefig(output_name)

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

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
         ' and Results/'
         )

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

print(Sector_Perf)

#Save the performance for later use

Sector_Performance = 'Performance '+test_sector

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Sector Performance/'
    )
output_name = path_to_file + Sector_Performance + '.csv'
Sector_Perf.to_csv(output_name)

print(Sector_Perf)

os.system('say "your program has finished"')
#%%

#Combining the Sector Portfolios

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Sector Performance/'
    )
os.chdir(path_to_file)
os.getcwd()

tickers_df1.sector.unique() #unique sectors

Healthcare_returns = pd.read_csv('Sector Returns Healthcare.csv', \
                                 parse_dates=True, index_col=[0])
Basic_Materials_returns = pd.read_csv('Sector Returns Basic Materials.csv', \
                                      parse_dates=True, index_col=[0])
Financial_Services_returns = pd.read_csv('Sector Returns Financial Services.csv', \
                                         parse_dates=True, index_col=[0])
Technology_returns = pd.read_csv('Sector Returns Technology.csv', \
                                 parse_dates=True, index_col=[0])
Industrials_returns = pd.read_csv('Sector Returns Industrials.csv', \
                                  parse_dates=True, index_col=[0])
Energy_returns = pd.read_csv('Sector Returns Energy.csv', \
                             parse_dates=True, index_col=[0])
Consumer_Cyclical_returns = pd.read_csv('Sector Returns Consumer Cyclical.csv', \
                                        parse_dates=True, index_col=[0])
Consumer_Defensive_returns = pd.read_csv('Sector Returns Consumer Defensive.csv', \
                                         parse_dates=True, index_col=[0])
Communication_Services_returns = pd.read_csv('Sector Returns Communication Services.csv', \
                                             parse_dates=True, index_col=[0])
Utilities_returns = pd.read_csv('Sector Returns Utilities.csv', \
                                parse_dates=True, index_col=[0])

merged_sectors = pd.concat([Healthcare_returns, Basic_Materials_returns,
                            Financial_Services_returns,
                            Technology_returns,
                            Industrials_returns,
                            Energy_returns,
                            Consumer_Cyclical_returns,
                            Consumer_Defensive_returns,
                            Communication_Services_returns,
                            Utilities_returns], axis=1, ignore_index=False)

equal_weight = merged_sectors.iloc[:, np.arange(0, 40, 4)]
equal_weight['EW portfolio'] = equal_weight.mean(axis=1)

expensive_port = merged_sectors.iloc[:, np.arange(2, 40, 4)]
expensive_port['Expensive portfolio'] = expensive_port.mean(axis=1)

cheap_port = merged_sectors.iloc[:, np.arange(1, 40, 4)]
cheap_port['Cheap portfolio'] = cheap_port.mean(axis=1)

Long_Short_port = merged_sectors.iloc[:, np.arange(3, 40, 4)]
Long_Short_port['Long Short portfolio'] = Long_Short_port.mean(axis=1)

#Import the Fama French Book Equity / Market Equity Value Portfolio as a Comparison
FF_value = web.DataReader('Portfolios_Formed_on_BE-ME_Daily', 'famafrench', start='1990-08-30',)

print(FF_value['DESCR'])

FF_value[0].head()

data = FF_value[0]
data = data.dropna()
data = data/100 #convert to percent returns
FF_value_data = data.copy()

FF_value_start_date = merged_sectors.first_valid_index()
FF_value_end_date = merged_sectors.last_valid_index()

FF_value_data = pd.DataFrame(FF_value_data['Hi 30'][FF_value_start_date:FF_value_end_date])
FF_value_data.columns = ['Fama French B/M Cheapest 30 percent']

####Import the Fama French Total Market Portfolio as a Comparison#############
FF_Market = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1990-08-30',)

print(FF_Market['DESCR'])

FF_Market[0].head()

data_market = FF_Market[0]
data_market = data_market.dropna()
data_market = data_market/100 #convert to percent returns
FF_Market_data = data_market.copy()
FF_Market_data['Mkt'] = FF_Market_data['Mkt-RF']+FF_Market_data['RF']
RF_data = (1+FF_Market_data['RF']).cumprod()
RF_data.columns = ['Risk Free']

FF_Market_start_date = merged_sectors.first_valid_index()
FF_Market_end_date = merged_sectors.last_valid_index()

FF_Market_data = pd.DataFrame(FF_Market_data['Mkt'][FF_Market_start_date:FF_Market_end_date])
FF_Market_data.columns = ['Fama French Total Market']

Portfolios = pd.concat([equal_weight['EW portfolio'],
                        expensive_port['Expensive portfolio'],
                        cheap_port['Cheap portfolio'],
                        Long_Short_port['Long Short portfolio'],
                        FF_value_data,
                        FF_Market_data], axis=1, ignore_index=False)

#Save the Portfolio Returns and Performance Chart for later use
Portfolio_Returns = 'Portfolio Returns'

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Sector Performance/'
    )
output_name = path_to_file + Portfolio_Returns + '.csv'
Portfolios.to_csv(output_name)

portfolio_index = (1 + Portfolios).cumprod()
ax = portfolio_index.plot(title='In Sample Portfolio Performance')
fig = ax.get_figure()
Sector_Performance_Chart = 'In Sample Portfolio Performance Chart'
path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Sector Performance/'
    )
output_name = path_to_file + Sector_Performance_Chart + '.png'
fig.savefig(output_name)

####    Calculate Risk and Returns for the Portfolio   #####

RF_start_date = portfolio_index.first_valid_index()
RF_end_date = portfolio_index.last_valid_index()

RF_data = pd.DataFrame(RF_data[RF_start_date:RF_end_date])
RF_Ann_Return_df = annualized_return(RF_data)
RF_Ann_Return = np.round(float(RF_Ann_Return_df.iloc[:, 1]), 4)

returns = annualized_return(portfolio_index)
Stddev = annualized_standard_deviation(portfolio_index)
Portfolio_Perf = returns.merge(Stddev)

Sharpe_Ratios = sharpe_ratio(portfolio_index, RF_Ann_Return)
Portfolio_Perf = Portfolio_Perf.merge(Sharpe_Ratios)

Sortino_Ratios = sortino_ratio(portfolio_index, RF_Ann_Return)
Portfolio_Perf = Portfolio_Perf.merge(Sortino_Ratios)

Max_DD = max_drawdown(portfolio_index)
Portfolio_Perf = Portfolio_Perf.merge(Max_DD)

Calmar_Ratios = calmar_ratio(portfolio_index)
Portfolio_Perf = Portfolio_Perf.merge(Calmar_Ratios)

Gain_To_Pain = gain_to_pain_ratio(portfolio_index)
Portfolio_Perf = Portfolio_Perf.merge(Gain_To_Pain)

Portfolio_Performance = 'Portfolio Performance'

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Sector Performance/'
    )
output_name = path_to_file + Portfolio_Performance + '.csv'
Portfolio_Perf.to_csv(output_name)

print(Portfolio_Perf)
#%%

####    testing statistical significance of alpha ####

#Sample Size
N = Portfolios.shape[0]

#Calculate the variance to get the standard deviation
Portfolio_Alpha = Portfolios['Cheap portfolio']- Portfolios['Fama French B/M Cheapest 30 percent']
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
### You can see that after comparing there is not a meaningful difference between
#fama french value and machine learning combined value portfolio

ZEROS = [0] * N
cheap_array = np.array(Portfolios['Cheap portfolio'].dropna())
#for whatever reason don't have return data in portfolios from 2002-04-01
#and in order to get t test need same number of data points so removed one
#day from Fama Frenchd data
FF_cheap_array = \
    np.array(Portfolios['Fama French B/M Cheapest 30 percent'].drop\
             (pd.to_datetime('2002-04-01')))
np.isnan(cheap_array).any()
np.isnan(FF_cheap_array).any()
## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(cheap_array, FF_cheap_array)
print("t = " + str(t2))
print("p = " + str(p2/2)) #one sided t test
#%%
#####Testing Statistical Significance of L/S Portfolio#########

#Sample Size
N = Portfolios.shape[0]

#Calculate the variance to get the standard deviation
#For unbiased max likelihood estimate we have to divide the var by N-1,
#and therefore the parameter ddof = 1
var_factor = Portfolios['Long Short portfolio'].var(ddof=1)

## Calculate the t-statistics
t = (Portfolios['Long Short portfolio'].mean() - 0) / np.sqrt(var_factor/N)

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
t2, p2 = stats.ttest_ind(Portfolios['Long Short portfolio'], ZEROS)
print("t = " + str(t2))
print("p = " + str(p2/2)) #one sided t test
