#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:13:17 2020

@author: downey
"""
import datetime
import pickle
import pandas as pd
import quandl
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import yfinance as yf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

##############################################################################

#### Sentiment and Machine Learning for Portfolio ###

##############################################################################

pd.set_option('display.max_columns', 14)
pd.set_option('display.max_rows', 1000)
plt.style.use('ggplot')
quandl.ApiConfig.api_key = "key here"
#turn off pandas warning for index/slicing as copy warning
pd.options.mode.chained_assignment = None  # default='warn'
#%%
#######################Fundamental and Equity Prices##########################

#fundamental data
fundamental_data = (
    pd.read_csv('fundmental data')
    )

#import all of the equity price data from csv from Sharadar
equity_prices = (
    pd.read_csv('equity price data')
    )

#get ticker meta data
tickers_df = (
    pd.read_csv('ticker data',low_memory=False)
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

#filtered USA fundamental  and equity price data
USA_fundamentals = fundamental_data[fundamental_data['ticker'].isin(list_tickers)]
USA_equity_prices = equity_prices[equity_prices['ticker'].isin(list_tickers)]

#############Creating the Macroeconomic feature set#########################
#%%
start = datetime.datetime(1996, 1, 1)
end = datetime.datetime(2020, 5, 31)

ECON_Policy_Uncertainty_Index = pdr.get_data_fred('USEPUINDXD', start, end)
ECON_Policy_Uncertainty_Index.columns = ['Economic Policy Uncertainty Index']

SAHM_Rule = pdr.get_data_fred('SAHMCURRENT', start, end)
SAHM_Rule.columns = ['Sahm Rule']
#data comes out 1 month after the fact
SAHM_Rule['Sahm Rule (Lag)'] = SAHM_Rule['Sahm Rule'].shift(1)

Yale_Shiller_US_Confidence = quandl.get(["YALE/US_CONF_INDEX_VAL_INDIV", 
                                         "YALE/US_CONF_INDEX_VAL_INST"])
Yale_Shiller_US_Confidence = Yale_Shiller_US_Confidence[Yale_Shiller_US_Confidence.columns[::2]]
#data comes out two months after the fact
Yale_Shiller_US_Confidence = Yale_Shiller_US_Confidence.shift(2)

Yale_Shiller_US_Confidence.columns = ['US Shiller Valuation Index Indiv (Lag)',
                                       'US Shiller Valuation Index Inst (Lag)']

Univ_Michigan_Sentiment = pdr.get_data_fred('UMCSENT', start, end)
Univ_Michigan_Sentiment.columns = ['University of Michigan Consumer Sentiment']

#data comes out two months after the fact
Univ_Michigan_Sentiment['University of Michigan Consumer Sentiment (Lag)'] = Univ_Michigan_Sentiment['University of Michigan Consumer Sentiment'].shift(2)

Duke_cfo = (
    pd.read_csv('/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On'
                '/Sentiment Trading/historical_cfo_data_clean.csv', index_col = [0], parse_dates = True)
    )
Duke_cfo.head()
Duke_cfo.columns = ['Duke CFO Optimism']
#1 quarter report lag
Duke_cfo['Duke CFO Optimism Lag'] = Duke_cfo['Duke CFO Optimism'].shift(1)

#CBOE Put Call Ratio
Put_Call_Ratio = (
    pd.read_csv('/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On'
                '/Sentiment Trading/CBOE_PUT_CALL_filtered.csv', index_col = [0], parse_dates = True)
    )
Put_Call_Ratio.head()
Put_Call_Ratio.columns = ['CBOE Put Call Ratio']

#create spread between OAS of high yield and A rated
Credit_Spreads = pdr.get_data_fred(['BAMLH0A0HYM2', 'BAMLC0A3CA'], start, end)
Credit_Spreads['spread'] = Credit_Spreads['BAMLH0A0HYM2']-Credit_Spreads['BAMLC0A3CA']
Credit_Spreads.columns = ['High Yield OAS', 'Inv Grade A OAS', 'Credit Spread']

VIX = pdr.get_data_fred('VIXCLS', start, end)
VIX.columns = ['VIX']

AAII_data = quandl.get("AAII/AAII_SENTIMENT", start_date=start, end_date=end)
AAII_data.head() 
AAII_data2 = AAII_data[['Bullish','Neutral','Bearish']]
#one week report lag
AAII_data3 = AAII_data2.shift(1)

results = pd.DataFrame()
results = pd.concat([results, ECON_Policy_Uncertainty_Index], axis = 1, ignore_index=False)
results = pd.concat([results, SAHM_Rule['Sahm Rule (Lag)']], axis = 1, ignore_index=False)
results = pd.concat([results, Credit_Spreads['Credit Spread']], axis = 1, ignore_index=False)
results = pd.concat([results, Yale_Shiller_US_Confidence], axis = 1, ignore_index=False)
results = pd.concat([results, Univ_Michigan_Sentiment['University of Michigan Consumer Sentiment (Lag)']], axis = 1, ignore_index=False)
results = pd.concat([results, Duke_cfo['Duke CFO Optimism Lag']], axis = 1, ignore_index=False)
results = pd.concat([results, Put_Call_Ratio], axis = 1, ignore_index=False)
results = pd.concat([results, VIX], axis = 1, ignore_index=False)
results = pd.concat([results, AAII_data3], axis = 1, ignore_index=False)

results_2 = results.fillna(method='ffill')
results_2 = results_2.dropna()

#Create Market Turnover Metric (total shares traded in a day/total shares outstanding)
USA_equity_prices['date'] = pd.to_datetime(USA_equity_prices['date'])
USA_equity_prices2 = USA_equity_prices.set_index('date')

#####Filtering the data and creating In Sample and removing duplicates#####

#Choose dimension quarerly as reported 'ARQ'
fundamentals = USA_fundamentals[USA_fundamentals.dimension == 'ARQ']

# Converting to date index which will be needed when doing the ML training
fundamentals['calendardate'] = pd.to_datetime(fundamentals['calendardate'])

#Find data rows where fundamentals have been restated for previous quarter
#and we want to remove for backtesting since at the time you only have the first
#release of data and not subsequent revisions
duplicateRowsDF = fundamentals[fundamentals.duplicated(['ticker', 'calendardate'])]

print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')
fundamentals = fundamentals.drop_duplicates(subset = ['ticker', 'calendardate'],\
                                            keep = 'first')

duplicateRowsDF = fundamentals[fundamentals.duplicated(['ticker', 'calendardate'])]
print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')

####Creating Market Turnover#####
daily_stock_volume_wide = USA_equity_prices.pivot\
            (index='date', columns='ticker', values='volume')
daily_stock_volume_wide = daily_stock_volume_wide.fillna(0)

Share_Traded = daily_stock_volume_wide.sum(axis=1)
Share_Traded = pd.DataFrame(Share_Traded)

index_data = pd.to_datetime(Share_Traded.index)
Share_Traded = Share_Traded.set_index(index_data)
     
Total_shares_wide = fundamentals.pivot\
            (index='calendardate', columns='ticker', values='shareswa')
Total_shares_wide = Total_shares_wide.fillna(0)

#lag 1 quarter since can't act on shares outstanding until reported lag
Total_shares_wide_lag = Total_shares_wide.shift(1)

Total_shares_sum = Total_shares_wide_lag.sum(axis=1)
Total_shares_sum = pd.DataFrame(Total_shares_sum)
#need to change index to dattime in order to resample below

index_data = pd.to_datetime(Total_shares_sum.index)
Total_shares_sum = Total_shares_sum.set_index(index_data)

Volume = pd.DataFrame()
Volume = pd.concat([Volume, Share_Traded], axis = 1, ignore_index = False)
Volume = pd.concat([Volume, Total_shares_sum], axis = 1, ignore_index = False)

Volume_2= Volume.fillna(method='ffill')

Volume_2.columns = ['Daily Volume','Total Shares Outstanding']

Volume_2 = Volume_2.dropna()
Volume_2['MARKET TURNOVER'] = Volume_2['Daily Volume'] / Volume_2['Total Shares Outstanding']

Volume_2 = Volume_2.replace([np.inf, -np.inf], np.nan)
Volume_2 = Volume_2.dropna()

results_2 = pd.concat([results_2, Volume_2['MARKET TURNOVER']], axis=1, ignore_index=False)

results_2 = results_2.dropna()

### Creating Advance Decline Ratio

####Creating List of Stock Prices#####
daily_stock_prices_wide = USA_equity_prices.pivot\
            (index='date', columns='ticker', values='close')
daily_stock_prices_wide = daily_stock_prices_wide.fillna(0)

daily_stock_prices_wide_returns = daily_stock_prices_wide.pct_change()
Advance_Decline = pd.DataFrame()
Advance_Decline['Advance'] = daily_stock_prices_wide_returns.select_dtypes(include='float64').gt(0).sum(axis=1)
Advance_Decline['Decline'] = daily_stock_prices_wide_returns.select_dtypes(include='float64').lt(0).sum(axis=1)
Advance_Decline['Advance/Decline Ratio'] = Advance_Decline['Advance']/(Advance_Decline['Advance']+Advance_Decline['Decline'])

results_2 = pd.concat([results_2, Advance_Decline['Advance/Decline Ratio']], axis=1, ignore_index=False)
results_2 = results_2.dropna()
#%%
# #SP500 tickers
# table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# df = table[0]
# sp500_tickers = list(df['Symbol'])
# sp500_tickers_trim = sp500_tickers[:20]
# sp500_tickers_trim.sort()
# sp500_tickers_trim
#%%
####Downloading SPDR Sector ETFs ####

#testing alternative tickers
#Sector_tickers = sp500_tickers_trim
#Sector_tickers = ['AAPL','GE','F','AMZN','PG','VTR','XOM','T']
Sector_tickers = ['XLE','XLY','XLP','XLF','XLV','XLI','XLB','XLK','XLU']
Sector_tickers.sort() #sort so will align with downloaded data

data = yf.download(Sector_tickers, start="1999-01-01", end="2020-06-30")
#%%
number_tickers = len(Sector_tickers)

Sector_Adj_Close_Prices = data.iloc[:,0:number_tickers]

Sector_Open_Prices = data.iloc[:,4*number_tickers:5*number_tickers]

#Create the adjustment factor historically
for i,j in zip(Sector_tickers,range(1,number_tickers+1,1)):
    data[str(i) + '_Adjusted Factor'] = 1 + ( data.iloc[:,j+8] - data.iloc[:,j-1] ) / data.iloc[:,j-1]

Factor = data.iloc[:,data.shape[1]-len(Sector_tickers):]

Sector_Open_Prices/np.array(Factor)

Sector_Adj_Open_Prices = Sector_Open_Prices/np.array(Factor)

Sector_Adj_Open_Prices.columns = Sector_tickers
Sector_Adj_Open_Prices.columns = [str(col) + '_Open' for col in Sector_Adj_Open_Prices.columns]
Sector_Adj_Close_Prices.columns = Sector_tickers
Sector_Adj_Close_Prices.columns = [str(col) + '_Close' for col in Sector_Adj_Close_Prices.columns]

#Create Rolling Volatility Annualized

Sector_Returns = Sector_Adj_Close_Prices.pct_change().dropna()
Sector_Returns.columns = Sector_tickers

Sector_Returns.index = pd.to_datetime(Sector_Returns.index)
## rolling annualized Volatility

Lookback_range = range(2,23,1)

#Merge each ticker volatility
vol_dict = {}
for i in Lookback_range:
    Realized_Vol = pd.DataFrame()
    Realized_Vol = Sector_Returns.rolling(i).std() * (252 ** .5)
    Realized_Vol.columns = [str(col) + '_Vol' for col in Realized_Vol.columns]
    Realized_Vol.index = pd.to_datetime(Realized_Vol.index)
    vol_dict[i] = pd.DataFrame()
    vol_dict[i] = Realized_Vol

vol_dict

#Merge each ticker trend/price ratio
Lookback_range_trend = range(2,23,1)

trend_dict = {}
for i in Lookback_range_trend:
    Realized_Trend = pd.DataFrame()
    Realized_Trend = Sector_Adj_Close_Prices.ewm(com=i).mean()
    Realized_Trend = (Realized_Trend/Sector_Adj_Close_Prices)
    Realized_Trend.columns = [str(col) + '_SMA' for col in Realized_Trend.columns]
    Realized_Trend.index = pd.to_datetime(Realized_Trend.index)
    trend_dict[i] = pd.DataFrame()
    trend_dict[i] = Realized_Trend

trend_dict

#Create Rolling X day returns
adjusted_df = pd.concat([Sector_Adj_Open_Prices, Sector_Adj_Close_Prices], axis=1, ignore_index=False)

#create for loop of buy on open and sell on t+2 close
Number_days_to_hold = 3

Returns_Open_Close_n_days = pd.DataFrame()
Returns_Open_Close_n_days = pd.DataFrame(np.nan, index=adjusted_df.index, columns=Sector_tickers)

for j in range(0, len(Sector_tickers)):
    for i in range(0, len(adjusted_df)-2):
        Returns_Open_Close_n_days.iloc[i, j] = (adjusted_df.iloc[i+2,j+len(Sector_tickers)]/adjusted_df.iloc[i,j])-1

Returns_Open_Close_n_days.columns = Sector_tickers
Sector_Returns_forward_n_days = Returns_Open_Close_n_days.dropna()
Sector_Returns_forward_n_days.index = pd.to_datetime(Sector_Returns_forward_n_days.index)
Sector_Returns_forward_n_days.columns = [str(col) + '_' + str(Number_days_to_hold)+'day_return' for col in Sector_Returns_forward_n_days.columns]

##### Create Data for Each Ticker ####
results_3 = results_2.copy()

#merge volatility, trend, and other features and value for each ticker
for col in Sector_tickers:
    Sector_Returns_Vol = pd.DataFrame()
    for i in Lookback_range:
        Vol_DF = pd.DataFrame.from_dict(vol_dict[i])
        Vol_DF = pd.DataFrame(Vol_DF[str(col) + '_Vol'])
        Vol_DF.columns = [(str(col) + '_Vol' + str(i))]
        Sector_Returns_Vol = pd.concat([Sector_Returns_Vol, Vol_DF], axis=1, ignore_index=False)
    Sector_Returns_Trend = pd.DataFrame()
    for i in Lookback_range_trend:
        Trend_DF = pd.DataFrame.from_dict(trend_dict[i])
        Trend_DF
        Trend_DF = pd.DataFrame(Trend_DF[str(col) + '_Close_SMA'])
        Trend_DF.columns = [(str(col) + '_SMA' + str(i))]
        Sector_Returns_Trend = pd.concat([Sector_Returns_Trend, Trend_DF], axis=1, ignore_index=False)
    Sector_Returns_Vol = pd.concat([Sector_Returns_Vol, Sector_Returns_Trend], axis=1, ignore_index=False)
    Sector_Returns_Vol = pd.concat([Sector_Returns_Vol, Sector_Returns_forward_n_days[str(col) + '_' + str(Number_days_to_hold)+'day_return']], axis=1, ignore_index=False)
    Sector_Returns_Vol.head()
    results_3 = pd.concat([results_3, Sector_Returns_Vol], axis = 1, ignore_index=False)
    #create the lagged returns we are trying to forecast. Go long at the end of t+1 signal and end position on close of t+4
    results_3[str(col) + '_' + str(Number_days_to_hold)+'day_return_lag'] = results_3[str(col) + '_' + str(Number_days_to_hold)+'day_return'].shift(-1)
    results_3['Up_or_Down_' + str(col)] = \
        np.where(results_3[str(col) + '_' + str(Number_days_to_hold)+'day_return_lag'] > .00, 1,0)
    results_3.rename(columns={'Up_or_Down_' + str(col):col}, inplace=True)

results_9 = results_3.reset_index()
results_9.rename(columns={'index':'date'}, inplace=True)
results_9.head()
feature_labels = list(results_9.columns[0:15])
tickers = Sector_tickers

#Make Wide to long
joineddata = [feature_labels + tickers]
long_data= pd.melt(results_9[feature_labels + tickers],
                    id_vars = feature_labels, 
            value_vars=tickers)

#Merge each ticker volatility and trend into pandas df from dictionary
d = {}
for name in tickers:
    Vol_Data_Sector = pd.DataFrame()
    for i in Lookback_range:
        vole_data = results_9[['date',(str(name)+ '_Vol' + str(i))]]
        vole_data = vole_data.set_index('date')
        vole_data.rename(columns={(str(name) + '_Vol' + str(i)):'Volatility_'+str(i)}, inplace=True)
        Vol_Data_Sector = pd.concat([Vol_Data_Sector, vole_data], axis=1, ignore_index=False)
    for i in Lookback_range_trend:
        trend_data = results_9[['date',(str(name)+ '_SMA' + str(i))]]
        trend_data = trend_data.set_index('date')
        trend_data.rename(columns={(str(name) + '_SMA' + str(i)):'SMA_'+str(i)}, inplace=True)
        Vol_Data_Sector = pd.concat([Vol_Data_Sector, trend_data], axis=1, ignore_index=False) 
    Vol_Data_Sector['variable'] = name
    d[name] = pd.DataFrame()
    d[name] = long_data.merge(Vol_Data_Sector, on = ['date','variable'])

DF11 = pd.DataFrame()

for i in tickers:
    DF11 = pd.concat([DF11,pd.DataFrame.from_dict(d[i])], axis=0, ignore_index=True)

DF11 = DF11.dropna()

#Convert Category tickers to numerical array for features

from sklearn.preprocessing import OneHotEncoder

ndf = DF11[['variable','VIX']]
enc = OneHotEncoder()
enc.fit(ndf)

new_array = enc.transform(ndf).toarray()
new_array.shape
#select only the first columns with ticker info
new_array = new_array[:,0:len(tickers)]

sectors = pd.DataFrame(new_array)
sectors.columns = tickers

DF11 = DF11.reset_index()

DF12 = pd.concat([DF11,sectors], axis = 1, ignore_index=False)

DF12 = DF12.drop(columns=['variable'])

results_5 = DF12.set_index('date')

Out_of_Sample_Start = '2014-01-02'

End_Date = results_5.index[-1]
End_Date = End_Date.strftime('%Y-%m-%d')

Out_of_Sample_End = End_Date

Out_of_Sample_Data = results_5[Out_of_Sample_Start:Out_of_Sample_End]

#sort the data by dates so it makes sense with train and validation
Out_of_Sample_Data_Sorted = Out_of_Sample_Data.sort_index()
index_data = Out_of_Sample_Data_Sorted.index
index_data = index_data.drop_duplicates()

#keep only every nth day so as to have independent return samples
Out_of_Sample_Data_Sorted = Out_of_Sample_Data_Sorted.loc[index_data[::Number_days_to_hold]]

features = Out_of_Sample_Data_Sorted.copy()
#%%
###########################Creating Training/Validation Set###################

# Labels are the values we want to predict
test_labels = np.array(features['value'])
# Remove the labels from the feature and the date
features_array = features.drop(['value','index'], axis=1)
# Saving feature names for later use
feature_list = list(features_array.columns)
# Convert to numpy array
test_features = np.array(features_array)
# Using Skicit-learn to split data into training and testing sets, but in this case
#it is a validation set
#%%

print('Test Features Shape:', test_features.shape)
print('Test Labels Shape:', test_labels.shape)

number_features = test_features.shape[1]
#%%

#Load the 5 ML Models

path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/ML Models/')
Data_Title = 'RF Model'
filename = path_to_file + Data_Title + '.sav'
rf = pickle.load(open(filename, 'rb'))

Data_Title = 'GB Model'
filename = path_to_file + Data_Title + '.sav'
GB = pickle.load(open(filename, 'rb'))

Data_Title = 'Neigh Model'
filename = path_to_file + Data_Title + '.sav'
Neigh = pickle.load(open(filename, 'rb'))

Data_Title = 'ABoost Model'
filename = path_to_file + Data_Title + '.sav'
ABoost = pickle.load(open(filename, 'rb'))

Data_Title = 'GP Model'
filename = path_to_file + Data_Title + '.sav'
GP = pickle.load(open(filename, 'rb'))
#%%
#### Create Predictions from Ensemble of acurate algos ####
ML_Model = rf
rf_predictions_test = ML_Model.predict(test_features)
rf_predictions_test = ML_Model.predict(test_features)

RF_PREDICTIONS = np.concatenate([rf_predictions_test, rf_predictions_test])

ML_Model = GB
gb_predictions_test = ML_Model.predict(test_features)
gb_predictions_test = ML_Model.predict(test_features)

GB_PREDICTIONS = np.concatenate([gb_predictions_test, gb_predictions_test])

ML_Model = Neigh
neigh_predictions_test = ML_Model.predict(test_features)
neigh_predictions_test = ML_Model.predict(test_features)

NEIGH_PREDICTIONS = np.concatenate([neigh_predictions_test, neigh_predictions_test])

ML_Model = ABoost
adaboost_predictions_test = ML_Model.predict(test_features)
adaboost_predictions_test = ML_Model.predict(test_features)

ADBOOST_PREDICTIONS = np.concatenate([adaboost_predictions_test, adaboost_predictions_test])

ML_Model = GP
gp_predictions_test = ML_Model.predict(test_features)
gp_predictions_test = ML_Model.predict(test_features)

GP_PREDICTIONS = np.concatenate([gp_predictions_test, gp_predictions_test])

PREDICTIONS = pd.DataFrame(np.column_stack((RF_PREDICTIONS, GB_PREDICTIONS, \
                                            NEIGH_PREDICTIONS, ADBOOST_PREDICTIONS, GP_PREDICTIONS)))
PREDICTIONS.columns = ['Random Forest', 'Gradient Boosting', 'K Nearest Neighbors', 'Adaboost', 'Gaussian Process']
#%%

#At least 3 out of 5 algos voting positive, otherwise go with 0
PREDICTIONS['Vote'] = np.where(PREDICTIONS.mean(axis=1)>.51,1,0)

PREDICTIONS_test = pd.DataFrame(np.column_stack\
                               ((rf_predictions_test, gb_predictions_test,\
                                 neigh_predictions_test, adaboost_predictions_test, gp_predictions_test)))
PREDICTIONS_test['Vote'] = np.round(PREDICTIONS_test.mean(axis=1),0)

accuracy_score(test_labels, np.array(PREDICTIONS_test['Vote']))
confusion_matrix(test_labels, np.array(PREDICTIONS_test['Vote']))
f1_score(test_labels, np.array(PREDICTIONS_test['Vote']))
roc_auc_score(test_labels, np.array(PREDICTIONS_test['Vote']))
print(classification_report(test_labels, np.array(PREDICTIONS_test['Vote'])))
#%%
features['Prediction'] = PREDICTIONS_test['Vote'].values

#########  Portfolio Construction  ########
Predictions_DF = features.iloc[:,(-len(tickers)-1):]

#Using the same in sample dates here and for equal weight benchmark
f_date = datetime.datetime.strptime(Out_of_Sample_Start, '%Y-%m-%d')
l_date = datetime.datetime.strptime(Out_of_Sample_End,  '%Y-%m-%d')

delta = l_date - f_date

#Choose the number of periods (days in range / the forecasted return days)
period_delta = np.floor(delta.days/(Number_days_to_hold))
period_delta = int(period_delta)
first_period = Out_of_Sample_Start #using f_date

returns_df_portfolio = pd.DataFrame()
row_length = Predictions_DF.shape[0]
Portfolio_Turnover = pd.DataFrame()

for i in range(0, row_length, number_tickers):

    decision_date = datetime.datetime.strftime(Predictions_DF.index[i], '%Y-%m-%d')
    df = Predictions_DF.loc[decision_date]
    sector_matrix = np.array(df.iloc[:,:-1])
    prediction_vector = np.array(df.iloc[:,-1])

    new_df = pd.DataFrame(sector_matrix.T * prediction_vector).T
    new_df.columns = tickers
    
    #get either 1 or 0 for weight and make equal weight
    weights = new_df.max()/len(tickers)
    #weights[weights == 0] = -1/len(tickers) #if you want short option
    
    #keep track of stocks and turnover
    Turnover = pd.DataFrame({'Date':decision_date,
                                 'Tickers':tickers,
                                 'Weight':weights})
    Portfolio_Turnover = Portfolio_Turnover.append(Turnover)
    
    #buy on the open of the next day (t) and close out on (t+n) 
    rolling_real_implementation = Returns_Open_Close_n_days.shift(-1)
    rolling_real_implementation.loc[decision_date]
    
    rolling_real_implementation.loc[decision_date].dot(weights)
    
    actual_returns = rolling_real_implementation.loc[decision_date].dot(weights)
    end_trade_date = pd.to_datetime(decision_date) + pd.tseries.offsets.BusinessDay(Number_days_to_hold)
    end_trade_date_trim = datetime.datetime.strftime(end_trade_date, '%Y-%m-%d')
    
    return_info = {'Date': [end_trade_date_trim],
            'Value': [actual_returns]
            }
    
    returns_df_portfolio2 = pd.DataFrame(return_info, columns = ['Date', 'Value'])  
    returns_df_portfolio = returns_df_portfolio.append(returns_df_portfolio2)

returns_df_portfolio['Date'] = pd.to_datetime(returns_df_portfolio['Date'])

returns_df_portfolio_2 = returns_df_portfolio.set_index('Date')

#Counting trades executed
Portfolio_Turnover['Trades'] = np.nan
for i in range(0, Portfolio_Turnover.shape[0]-number_tickers,1):
        Portfolio_Turnover.iloc[i+number_tickers,3] = \
            np.where((Portfolio_Turnover.iloc[i,2] == \
                      Portfolio_Turnover.iloc[(i+number_tickers),2]),0,1)
Total_Trades = Portfolio_Turnover['Trades'].sum()
print("There were " + str(int(Total_Trades)) + " trades.")

slippage = .0015 #assume 15bps one way transaction, and $1 for trade
trade_commission = 1
gross_fees = (Total_Trades * slippage * 1/number_tickers)

print("After market impact, you would deduct " + \
      str(np.round(gross_fees*100,2)) + \
          "% from gross performance and " + str(Total_Trades*trade_commission) + " dollars.")
    
####   Equal Weight Portfolio ###

Equal_Weight_Returns =  Sector_Adj_Close_Prices[Out_of_Sample_Start:Out_of_Sample_End]
Equal_Weight_Returns = Equal_Weight_Returns.pct_change().dropna()
Equal_Weight_Returns = Equal_Weight_Returns.div(len(tickers))
Equal_Weight_Returns['Equal Weight Returns'] = Equal_Weight_Returns.sum(axis=1)
Equal_Weight_Returns['Equal Weight Index'] = (1 + Equal_Weight_Returns['Equal Weight Returns']).cumprod()
#%%
portfolio_index = (1 + returns_df_portfolio_2).cumprod()
portfolio_index.columns = ['Sentiment Strategy']
plt.rcParams.update(plt.rcParamsDefault)
ax = portfolio_index.plot(title='Sentiment Out of Sample performance', logy=False)
fig = ax.get_figure()
Performance_Chart = 'Sentiment Out of Sample Performance Chart'
path_to_file = (
    r'/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On/Sentiment Trading/Visuals/'
        )
output_name = path_to_file + Performance_Chart + '.pdf'
fig.savefig(output_name)
#%%
##### Performance Stats ####

results_performance = pd.DataFrame()
D = len(portfolio_index)
perf = ((portfolio_index.iloc[-1,-1])**((252/Number_days_to_hold)/D))-1
perf_realistic = ((portfolio_index.iloc[-1,-1]-gross_fees)**((252/Number_days_to_hold)/D))-1
print("After costs, the realistic annualized return would be " + str(np.round(perf_realistic,3)) + '%.')
std = portfolio_index.pct_change().std() * (252/Number_days_to_hold) ** 0.5
Sharpe = perf/std
roll_max = portfolio_index.expanding().max()
daily_drawdown = portfolio_index/roll_max - 1.0
# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window
max_daily_drawdown = daily_drawdown.expanding().min()
# Plot the results
#Daily_Drawdown.plot()
#Max_Daily_Drawdown.plot()
max_dd = float(np.min(max_daily_drawdown))
results_performance = results_performance.append(pd.DataFrame(
                {'MODEL': 'Strategy',
                 'SHARPE': Sharpe[-1],
                 'STD': std[-1],
                 'RETURN': perf,
                 'MAX DD': max_dd},
                 index = [0]), ignore_index = True)


Out_of_Sample_Start_Date_2 = portfolio_index.first_valid_index() + pd.tseries.offsets.BusinessDay(-1)
Equal_Weight_Returns_Out_of_Sample = Equal_Weight_Returns['Equal Weight Returns'][Out_of_Sample_Start_Date_2:]
Equal_Weight_Returns_Out_of_Sample.head()

EW_Returns_Out_of_Sample = Equal_Weight_Returns_Out_of_Sample
EW_Returns_Out_of_Sample_Reset_Index = (1+EW_Returns_Out_of_Sample).cumprod()
results_performance_eq = pd.DataFrame()
D = len(EW_Returns_Out_of_Sample_Reset_Index)
perf = EW_Returns_Out_of_Sample_Reset_Index.iloc[-1,]**(252/D)-1
std = EW_Returns_Out_of_Sample_Reset_Index.pct_change().std() * (252) ** 0.5
Sharpe = perf/std
roll_max = EW_Returns_Out_of_Sample_Reset_Index.expanding().max()
daily_drawdown = EW_Returns_Out_of_Sample_Reset_Index/roll_max - 1.0
# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window
max_daily_drawdown = daily_drawdown.expanding().min()
# Plot the results
#Daily_Drawdown.plot()
#Max_Daily_Drawdown.plot()
max_dd = float(np.min(max_daily_drawdown))
results_performance_eq = results_performance_eq.append(pd.DataFrame(
                {'MODEL': 'Equal Weight',
                 'SHARPE': Sharpe,
                 'STD': std,
                 'RETURN': perf,
                 'MAX DD': max_dd},
                 index = [0]), ignore_index = True)

performance_combo = pd.concat([results_performance, results_performance_eq], axis = 0, ignore_index=True)
print(performance_combo)

Portfolio_Performance = 'Out of Sample Portfolio Performance'

path_to_file = (r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/'
                )
output_name = path_to_file + Portfolio_Performance + '.csv'
performance_combo.to_csv(output_name)

#merge the strategy and equal weight indices
performance_results = pd.DataFrame()
performance_results = pd.concat([performance_results, portfolio_index], axis = 1, ignore_index=False)
performance_results = pd.concat([performance_results, EW_Returns_Out_of_Sample_Reset_Index], axis = 1, ignore_index=False)
performance_results = performance_results.fillna(method='ffill').fillna(value=1.0)
performance_results.iloc[0,0:2] = 1 #set initial value as 1
performance_results.tail()

ax = performance_results.plot(title='Out of Sample performance')
fig = ax.get_figure()
Performance_Chart = 'Out of Sample Performance Chart'
path_to_file = (
    r'/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On/Sentiment Trading/Visuals/'
        )
output_name = path_to_file + Performance_Chart + '.pdf'
fig.savefig(output_name)
#%%
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 7))
ax.set_title('Model Confusion Matrix')
sns.heatmap(confusion_matrix(test_labels, np.array(PREDICTIONS_test['Vote'])), annot=True, fmt='d')

fig = ax.get_figure()
Chart = 'Model Confusion Matrix'
path_to_file = (
    r'/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On/Sentiment Trading/Visuals/'
        )
output_name = path_to_file + Chart + '.pdf'
fig.savefig(output_name)

#Model Statistics

Models = ['rf','GB','ABoost','Neigh','GP']
Model_Stats = pd.DataFrame()

for i in Models:
    ml_model = eval(i)
    accuracy = accuracy_score(test_labels, ml_model.predict(test_features))
    f1 = f1_score(test_labels, ml_model.predict(test_features))
    roc = roc_auc_score(test_labels, ml_model.predict(test_features))
    recall = recall_score(test_labels, ml_model.predict(test_features))
    precision = precision_score(test_labels, ml_model.predict(test_features))
    brier = brier_score_loss(test_labels, ml_model.predict(test_features))
    Model_Stats = Model_Stats.append(pd.DataFrame(
                {'Model': str(i),
                 'Accuracy': accuracy,
                 'F1 Score': f1,
                 'ROC Area Under Curve': roc,
                 'Recall': recall,
                 'Precision': precision,
                 'Brier Score': brier},
                 index = [0]), ignore_index = True)

accuracy_ensemble = accuracy_score(test_labels, np.array(PREDICTIONS_test['Vote']))
f1_ensemble = f1_score(test_labels, np.array(PREDICTIONS_test['Vote']))
roc_ensemble = roc_auc_score(test_labels, np.array(PREDICTIONS_test['Vote']))
recall_ensemble = recall_score(test_labels, np.array(PREDICTIONS_test['Vote']))
precision_ensemble = precision_score(test_labels, np.array(PREDICTIONS_test['Vote']))
brier_ensemble = brier_score_loss(test_labels, np.array(PREDICTIONS_test['Vote']))
print(classification_report(test_labels, np.array(PREDICTIONS_test['Vote'])))

Model_Stats = Model_Stats.append(pd.DataFrame(
                {'Model': 'Ensemble',
                 'Accuracy': accuracy_ensemble,
                 'F1 Score': f1_ensemble,
                 'ROC Area Under Curve': roc_ensemble,
                 'Recall': recall_ensemble,
                 'Precision': precision_ensemble,
                 'Brier Score': brier_ensemble},
                 index = [0]), ignore_index = True) 

n = len(test_labels)
market_up = np.repeat(1,n)

accuracy_naive = accuracy_score(test_labels, market_up)
f1_naive = f1_score(test_labels, market_up)
roc_naive = roc_auc_score(test_labels, market_up)
recall_naive = recall_score(test_labels, market_up)
precision_naive = precision_score(test_labels, market_up)
brier_naive = brier_score_loss(test_labels, market_up)

Model_Stats = Model_Stats.append(pd.DataFrame(
                {'Model': 'Markets Always Up',
                 'Accuracy': accuracy_naive,
                 'F1 Score': f1_naive,
                 'ROC Area Under Curve': roc_naive,
                 'Recall': recall_naive,
                 'Precision': precision_naive,
                 'Brier Score': brier_naive},
                 index = [0]), ignore_index = True)

print(Model_Stats)

Model_Performance = 'Model Performance Out of Sample'

path_to_file = (r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/'
                )
output_name = path_to_file + Model_Performance + '.csv'
Model_Stats.to_csv(output_name)
#%%
