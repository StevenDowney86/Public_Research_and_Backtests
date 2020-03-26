#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:01:14 2020

@author: downey
"""
import datetime
import os
import warnings
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

#### This script combines the 4 value factors from the sector Random Forests
#### models and then chooses the stocks with the best value score based on 
#### that forecast. The resulting portfolio is a combination of the sector 
#### portfolios

##############################################################################

pd.set_option('display.max_columns', 110)
pd.set_option('display.max_rows', 1000)
plt.style.use('seaborn')
warnings.filterwarnings("ignore")
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
test_sector = 'Healthcare' #Include Utilities, Consumer Defensive, Industrials,
#Basic Materials, and Healthcare due to low MAE variance and high number of firms

#import the forecasted values for the sector

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
         ' and Results/Random Forest Models/ROIC Forecasts/'
         )
Data_for_Portfolio_ROIC = pd.read_csv(str(test_sector) + \
                                      '_OOS_ROIC_Predictions.csv', index_col=[0])

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
         ' and Results/Random Forest Models/PE Forecasts/'
         )
Data_for_Portfolio_PE = pd.read_csv(str(test_sector) + \
                                    '_OOS_PE_Predictions.csv', index_col=[0])

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
         ' and Results/Random Forest Models/EVEBITDA Forecasts/'
         )
Data_for_Portfolio_EVEBITDA = pd.read_csv(str(test_sector) + \
                                          '_OOS_EVEBITDA_Predictions.csv', index_col=[0])

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
         ' and Results/Random Forest Models/PFCF Forecasts/'
         )
Data_for_Portfolio_PFCF = pd.read_csv(str(test_sector) + \
                                      '_OOS_PFCF_Predictions.csv', index_col=[0])

#merge the data
Data_for_Portfolio = Data_for_Portfolio_ROIC.merge(Data_for_Portfolio_PE)
Data_for_Portfolio = Data_for_Portfolio.merge(Data_for_Portfolio_EVEBITDA)
Data_for_Portfolio = Data_for_Portfolio.merge(Data_for_Portfolio_PFCF)

#put tickers of Energy in a list
tickers = Data_for_Portfolio['ticker'].tolist()
len(tickers)

#sort out Sector Prices
Sector_stock_prices = equity_prices.loc  \
    [equity_prices['ticker'].isin(tickers)]

#testing Out of Sample Data
f_date = datetime.date(2014, 3, 31)#must have at least 4 quarters after the
#first available fundamental data point. This will equate to the f_date + 4 quarters
#due to the tranche approach + trading 1 quarter after fundamentals available
#and waiting until have data on all four tranches. So with a f_date = 2000-06-30,
#the first date for returns will be 2001-07-03

l_date = datetime.date(2019, 6, 30)#choosing the last date, results in last
#date for returns is l_date + 1 quarter

delta = l_date - f_date
quarters_delta = np.floor(delta.days/(365/4))-4
quarters_delta = int(quarters_delta)
first_quarter = str('2014-03-31')
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

        #filter the data for only current date to look at
        Date = pd.to_datetime(first_quarter_adapted) + pd.tseries.offsets.QuarterEnd(i)
        Date = Date.strftime('%Y-%m-%d')
        Data_for_Portfolio_master_filter = \
            Data_for_Portfolio_master.loc[Data_for_Portfolio_master['calendardate'] == Date]

        #Winsorize the metric data and compress outliers if desired
        Data_for_Portfolio_master_filter['Forward ROIC Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter['Forward ROIC'],\
                                   limits=Winsorize_Threshold)
        Data_for_Portfolio_master_filter['Forward E/P Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter['Forward E/P'],\
                                   limits=Winsorize_Threshold)
        Data_for_Portfolio_master_filter['Forward EBITDA/EV Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter['Forward EBITDA/EV'],\
                                   limits=Winsorize_Threshold)
        Data_for_Portfolio_master_filter['Forward FCF/P Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter['Forward FCF/P'],\
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

        left_over_firms = number_firms - firms_in_percentile
        Data_for_Portfolio_master_filter = \
            Data_for_Portfolio_master_filter.sort_values('Valuation Score', ascending=False)

        #filter the dataset by the percentile for expensive and cheap
        Sector_stocks_cheapest = Data_for_Portfolio_master_filter.iloc[:int(firms_in_percentile)]
        Sector_stocks_expensive = Data_for_Portfolio_master_filter.iloc[int(-left_over_firms):]

        #convert the list of unique tickers to a list
        Sector_stocks_cheapest_tickers = Sector_stocks_cheapest['ticker'].tolist()
        Sector_stocks_expensive_tickers = Sector_stocks_expensive['ticker'].tolist()

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

        ####    Cheap Stock Price Returns and Portfolio   ####

        #pick out start date and end date for calculating equity returns
        Sector_stock_prices_cheapest_wide.loc[start_date:end_date].head()
        Sector_stock_prices_cheapest_wide = \
            Sector_stock_prices_cheapest_wide.loc[start_date:end_date]

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

    portfolio_returns.columns = [('Cheap Tranche ' + str(Tranche)), \
                                 ('Expensive Tranch ' + str(Tranche)), 'LS']
    portfolio_tranche_Expensive_returns = \
        pd.concat([portfolio_tranche_Expensive_returns, portfolio_returns.iloc[:, 1:2]],\
                  axis=1, ignore_index=False)
    portfolio_tranche_Cheap_returns = \
        pd.concat([portfolio_tranche_Cheap_returns, portfolio_returns.iloc[:, 0]],\
                  axis=1, ignore_index=False)

#trim the data where we have returns for all tranches
first_date = portfolio_tranche_Cheap_returns.iloc[:, -1].first_valid_index()
last_date = portfolio_tranche_Cheap_returns.iloc[:, 0].last_valid_index()

condensed_Cheap_portfolio = pd.DataFrame()
condensed_Cheap_portfolio = portfolio_tranche_Cheap_returns[first_date:last_date]

condensed_Expensive_portfolio = pd.DataFrame()
condensed_Expensive_portfolio = portfolio_tranche_Expensive_returns[first_date:last_date]

#Remove rows with NaN's because we want to take equal weight of all 4 portfolios
#when they have data
condensed_Cheap_portfolio = pd.DataFrame(condensed_Cheap_portfolio.dropna(axis='rows'))
condensed_Expensive_portfolio = pd.DataFrame(condensed_Expensive_portfolio.dropna(axis='rows'))
#Create a Portfolio by averaging the returns from all 4 tranches
condensed_Cheap_portfolio['Combined Cheap Tranche Portfolio'] = \
    condensed_Cheap_portfolio.mean(axis=1)
condensed_Expensive_portfolio['Combined Expensive Tranche Portfolio'] = \
    condensed_Expensive_portfolio.mean(axis=1)

#Combine into one DataFrame
Combo = pd.concat([condensed_Cheap_portfolio['Combined Cheap Tranche Portfolio'],\
                   condensed_Expensive_portfolio['Combined Expensive Tranche Portfolio']],\
                  axis=1, ignore_index=False)

#Equal Weight Portfolio as Benchmark
f_date = datetime.date(2014, 3, 31)
l_date = datetime.date(2019, 6, 30)
delta = l_date - f_date
quarters_delta = np.floor(delta.days/(365/4))
quarters_delta = int(quarters_delta)
first_quarter = str('2014-03-31')
equal_weight_returns = pd.DataFrame()
Data_for_Portfolio_master = Data_for_Portfolio

for i in range(0, quarters_delta, 1):

    Date = pd.to_datetime(first_quarter) + pd.tseries.offsets.QuarterEnd(i)
    Date = Date.strftime('%Y-%m-%d')

    Data_for_Portfolio_master_filter = \
        Data_for_Portfolio_master.loc[Data_for_Portfolio_master['calendardate'] == Date]
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
    Sector_stock_prices_wide.head()
    returns_daily = Sector_stock_prices_wide.pct_change()
    returns_daily = returns_daily.dropna(how='all') #get rid of first NaN row
    returns_daily = returns_daily.dropna(axis='columns')
    column_length = np.where(returns_daily.shape[1] == 0, 1, returns_daily.shape[1]) #if no column

    portfolio_weights = np.repeat(1/column_length, column_length)
    returns_daily['portfolio return'] = returns_daily.dot(portfolio_weights)
    EW_returns = returns_daily['portfolio return']
    EW_returns_df = pd.DataFrame(EW_returns)
    equal_weight_returns = equal_weight_returns.append(EW_returns_df)

#change inf to NAs since first entry is inf
equal_weight_returns = equal_weight_returns.replace([np.inf, -np.inf], 0)

Combined_Portfolio_returns = pd.merge(equal_weight_returns, Combo,\
                                      how='inner', left_index=True, right_index=True)

#change inf to NAs since first entry is inf
Combined_Portfolio_returns = Combined_Portfolio_returns.replace([np.inf, -np.inf], 0)

#Rename the columns
Combined_Portfolio_returns = Combined_Portfolio_returns.rename\
    (columns={"portfolio return": "Equal Weight"})

Combined_Portfolio_returns['Long / Short'] = \
    Combined_Portfolio_returns['Combined Cheap Tranche Portfolio'] - \
        Combined_Portfolio_returns['Combined Expensive Tranche Portfolio']

#Save the Sector Returns to csv for use later
Sector_Returns = 'OOS Sector Returns '+test_sector

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, '
    'Code, and Results/Sector Performance/'
    )
output_name = path_to_file + Sector_Returns + '.csv'
Combined_Portfolio_returns.to_csv(output_name)

#create a performance chart and save for later
portfolio_index = (1 + Combined_Portfolio_returns).cumprod()
ax = portfolio_index.plot(title='In Sample ' + str(test_sector) + ' performance')
fig = ax.get_figure()
Sector_Performance_Chart = 'In Sample Performance Chart '+test_sector
path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Sector Performance/'
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

Sector_Performance = 'OOS_Performance '+test_sector

path_to_file = (r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models,'
                ' Code, and Results/Sector Performance/'
                )
output_name = path_to_file + Sector_Performance + '.csv'
Sector_Perf.to_csv(output_name)

os.system('say "your program has finished"')
#%%

#Combining the Sector Portfolios

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Sector Performance/'
    )
os.chdir(path_to_file)
os.getcwd()

Healthcare_returns = pd.read_csv('OOS Sector Returns Healthcare.csv', \
                                 index_col=[0], parse_dates=True)
Basic_Materials_returns = pd.read_csv('OOS Sector Returns Basic Materials.csv', \
                                      index_col=[0], parse_dates=True)
Industrials_returns = pd.read_csv('OOS Sector Returns Industrials.csv', \
                                  index_col=[0], parse_dates=True)
Consumer_Defensive_returns = pd.read_csv('OOS Sector Returns Consumer Defensive.csv',\
                                         index_col=[0], parse_dates=True)
Utilities_returns = pd.read_csv('OOS Sector Returns Utilities.csv', \
                                index_col=[0], parse_dates=True)

merged_sectors = pd.concat([Healthcare_returns, Basic_Materials_returns,
                            Industrials_returns,
                            Consumer_Defensive_returns,
                            Utilities_returns], axis=1, ignore_index=False)

column_number = merged_sectors.shape[1]

#rotate through every 4 column starting with column 0
equal_weight = merged_sectors.iloc[:, np.arange(0, column_number, 4)]
equal_weight['EW portfolio'] = equal_weight.mean(axis=1)

#rotate through every 4 column starting with column 1
expensive_port = merged_sectors.iloc[:, np.arange(1, column_number, 4)]
expensive_port['Expensive portfolio'] = expensive_port.mean(axis=1)

#rotate through every 4 column starting with column 2
cheap_port = merged_sectors.iloc[:, np.arange(2, column_number, 4)]
cheap_port['Cheap portfolio'] = cheap_port.mean(axis=1)

#create Long/Short Portfolios
Long_Short_port = pd.DataFrame()
Long_Short_port['L/S portfolio'] = \
    cheap_port['Cheap portfolio'] - expensive_port['Expensive portfolio']

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
FF_value_data.head()
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

#####Import Fama French Sectors used in Model################################
FF_Industries = web.DataReader('10_Industry_Portfolios_daily', 'famafrench', start='1990-08-30',)

print(FF_Industries['DESCR'])

FF_Industries[0].head()

data_Industries = FF_Industries[0]
data_Industries = data_Industries.dropna()
data_Industries = data_Industries/100 #convert to percent returns
FF_Industries_data = data_Industries.copy()
FF_Industries_data_benchmark = FF_Industries_data.drop(columns=['NoDur', 'HiTec', 'Telcm',\
                                                            'Enrgy', 'HiTec', 'Shops', 'Other'])
FF_Industries_data_benchmark.head()

FF_Industries_data_benchmark['Equal_Weight_Sector'] = FF_Industries_data_benchmark.mean(axis=1)
FF_Industries_data_benchmark['Equal_Weight_Sector'].head()
FF_Industries_start_date = merged_sectors.first_valid_index()
FF_Industries_end_date = merged_sectors.last_valid_index()

FF_Industries_data_benchmark_index = \
    pd.DataFrame(FF_Industries_data_benchmark['Equal_Weight_Sector']\
                 [FF_Industries_start_date:FF_Industries_end_date])
FF_Industries_data_benchmark_index.columns = ['Fama French Equal Weight Portfolio Sectors']
#########################Combine Portfolios and Indices#######################
Portfolios = pd.concat([equal_weight['EW portfolio'],
                        expensive_port['Expensive portfolio'],
                        cheap_port['Cheap portfolio'],
                        Long_Short_port['L/S portfolio'],
                        FF_value_data,
                        FF_Market_data,
                        FF_Industries_data_benchmark_index], axis=1, ignore_index=False)
#Save the Portfolio Returns and Performance Chart for later use
Portfolio_Returns = 'OOS Portfolio Returns'

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Sector Performance/'
    )
output_name = path_to_file + Portfolio_Returns + '.csv'
Portfolios.to_csv(output_name)

portfolio_index = (1 + Portfolios).cumprod()
ax = portfolio_index.plot(title='Out of Sample Portfolio Performance',
                          colormap='tab20b')
fig = ax.get_figure()
Sector_Performance_Chart = 'Out of Sample Portfolio Performance Chart'
path_to_file = (r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models,'
                ' Code, and Results/Sector Performance/'
                )
output_name = path_to_file + Sector_Performance_Chart + '.png'
fig.savefig(output_name)

#Compare to own data
ax = portfolio_index[['EW portfolio',
                      'Expensive portfolio',
                      'Cheap portfolio',
                      'L/S portfolio']].plot\
    (title='Out of Sample Portfolio Performance', colormap='tab20b')
fig = ax.get_figure()

#compare to Fama French Data
ax = portfolio_index[['Cheap portfolio',
                      'Fama French B/M Cheapest 30 percent',
                      'Fama French Total Market',
                      'Fama French Equal Weight Portfolio Sectors']].plot\
    (title='Out of Sample Portfolio Performance')
fig = ax.get_figure()

####    Calculate Risk and Returns for the Portfolio    ####

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

path_to_file = (r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models,'
                ' Code, and Results/Sector Performance/'
                )
output_name = path_to_file + Portfolio_Performance + '.csv'
Portfolio_Perf.to_csv(output_name)

print(Portfolio_Perf)
#%%
####    Performance of each sector vs. market cap    ####

Industrials_returns['FF Manufacturing'] = FF_Industries_data_benchmark['Manuf']
Industrials_index = (1 + Industrials_returns).cumprod()
ax = Industrials_index.plot(title='Out of Sample Industrials Portfolio Performance')
fig = ax.get_figure()

Basic_Materials_returns['FF Manufacturing'] = FF_Industries_data_benchmark['Manuf']
Basic_Materials_index = (1 + Basic_Materials_returns).cumprod()
ax = Basic_Materials_index.plot(title='Out of Sample Basic Materials Portfolio Performance')
fig = ax.get_figure()

Healthcare_returns['FF Healthcare'] = FF_Industries_data_benchmark['Hlth ']
Healthcare_index = (1 + Healthcare_returns).cumprod()
ax = Healthcare_index.plot(title='Out of Sample Healthcare Portfolio Performance')
fig = ax.get_figure()

Utilities_returns['FF Utilities'] = FF_Industries_data_benchmark['Utils']
Utilities_index = (1 + Utilities_returns).cumprod()
ax = Utilities_index.plot(title='Out of Sample Utilities Portfolio Performance')
fig = ax.get_figure()

Consumer_Defensive_returns['FF Durables'] = FF_Industries_data_benchmark['Durbl']
Consumer_Defensive_index = (1 + Consumer_Defensive_returns).cumprod()
ax = Consumer_Defensive_index.plot(title='Out of Sample Consumer Defensive Portfolio Performance')
fig = ax.get_figure()

#%%
####    Testing Statistical Significance of Alpha    ####
#Sample Size
N = Portfolios.shape[0]

#Calculate the variance to get the standard deviation
Portfolio_Alpha = Portfolios['Cheap portfolio']- Portfolios['EW portfolio']
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
### You can see that after comparing the
ZEROS = [0] * N

#convert to array
Portfolio_Alpha_array = np.array(Portfolio_Alpha)
ZEROS_array = np.array(ZEROS)

## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(Portfolio_Alpha_array, ZEROS_array, nan_policy='omit')
print("t = " + str(t2))
print("p = " + str(p2/2))
#%%
#####Testing Statistical Significance of L/S Portfolio#########

#Sample Size
N = Portfolios.shape[0]

#Calculate the variance to get the standard deviation
#For unbiased max likelihood estimate we have to divide the var by N-1,
#and therefore the parameter ddof = 1
var_factor = Portfolios['L/S portfolio'].var(ddof=1)

## Calculate the t-statistics
t = (Portfolios['L/S portfolio'].mean() - 0) / np.sqrt(var_factor/N)

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
t2, p2 = stats.ttest_ind(Portfolios['L/S portfolio'], ZEROS, nan_policy='omit')
print("t = " + str(t2))
print("p = " + str(p2/2)) #one sided t test

#statistically significant, more than the alpha of cheap - EW portfolio, but not
#at the 5% level
