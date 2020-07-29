#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:13:17 2020

@author: downey
"""
import os
import datetime
import time
import pickle
import pandas as pd
import quandl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web
import pandas_datareader as pdr
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

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
quandl.ApiConfig.api_key = "api key here"
#turn off pandas warning for index/slicing as copy warning
pd.options.mode.chained_assignment = None  # default='warn'
#%%
#######################Fundamental and Equity Prices##########################

#fundamental data
fundamental_data = (
    pd.read_csv('fundamental data')
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
#Remove df not needed to save memory

del fundamental_data
del USA_fundamentals
del USA_equity_prices
del USA_equity_prices2
del equity_prices
del fundamentals
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
Start_Date = results_5.index[0]
Start_Date = Start_Date.strftime('%Y-%m-%d')

In_Sample_Start = Start_Date
In_Sample_End = '2014-01-01'

In_Sample_Data = results_5[In_Sample_Start:In_Sample_End]

#sort the data by dates so it makes sense with train and validation
In_Sample_Data_Sorted = In_Sample_Data.sort_index()
index_data = In_Sample_Data_Sorted.index
index_data = index_data.drop_duplicates()

#keep only every nth day so as to have independent return samples
In_Sample_Filtered = In_Sample_Data_Sorted.loc[index_data[::Number_days_to_hold]]

features = In_Sample_Filtered.copy()
features.head()
#%%
###########################Creating Training/Validation Set###################

# Labels are the values we want to predict
labels = np.array(features['value'])
# Remove the labels from the feature and the date
features_array = features.drop(['value','index'], axis=1)
# Saving feature names for later use
feature_list = list(features_array.columns)
# Convert to numpy array
features_array = np.array(features_array)
# Using Skicit-learn to split data into training and testing sets, but in this case
#it is a validation set
#%%
test_ratio = .2 #variable for the percentage of train data to leave as validation

# Split the data into training and testing sets, with simple split
train_features, validation_features, train_labels, validation_labels = \
    train_test_split(features_array, labels, test_size=test_ratio, random_state=42, \
                     shuffle=False)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', validation_features.shape)
print('Testing Labels Shape:', validation_labels.shape)

number_features = train_features.shape[1]
#%%

##### SVC Linear #####

SVC_model = make_pipeline(StandardScaler(), \
                          LinearSVC(random_state=42, dual=False, tol=1e-5,\
                                    verbose=1,
                                    class_weight='balanced'))

SVC_model.fit(train_features, train_labels)

SVC_model.score(train_features, train_labels)

roc_auc_score(train_labels, SVC_model.predict(train_features))
confusion_matrix(train_labels, SVC_model.predict(train_features))
accuracy_score(train_labels, SVC_model.predict(train_features))
recall_score(train_labels, SVC_model.predict(train_features))
precision_score(train_labels, SVC_model.predict(train_features))

SVC_model.score(validation_features, validation_labels)
roc_auc_score(validation_labels, SVC_model.predict(validation_features))
confusion_matrix(validation_labels, SVC_model.predict(validation_features))
accuracy_score(validation_labels, SVC_model.predict(validation_features))
recall_score(validation_labels, SVC_model.predict(validation_features))
precision_score(validation_labels, SVC_model.predict(validation_features))
brier_score_loss(validation_labels, SVC_model.predict(validation_features))

print(classification_report(validation_labels, SVC_model.predict(validation_features)))
#%%

#### Create 5 K Fold Split Time Series ####

number_of_splits = 5
tscv = TimeSeriesSplit(n_splits=number_of_splits)
#%%
######Randomized CV for Random Forest using Timseries split #######
   
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
print(random_grid) 

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(n_jobs=-1)
# Random search of parameters, using time series split, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, \
                                n_iter = 100, cv = tscv, verbose=2, random_state=42, \
                                    scoring='f1')
# Fit the random search model
search_rf = rf_random.fit(train_features, train_labels)

search_rf.best_params_
search_rf.cv_results_
search_rf.best_score_
search_rf.best_estimator_
search_rf.scorer_

# save the CV model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')

Data_Title = 'RF CV Results'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(search_rf, open(filename, 'wb'))

### Load the RF Random CV results if needed #####
#%%
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')
Data_Title = 'RF CV Results'
filename = path_to_file + Data_Title + '.sav'
search_rf = pickle.load(open(filename, 'rb'))

scores_df_rf = pd.DataFrame(search_rf.cv_results_)
total_folds = np.linspace(1,number_of_splits,number_of_splits).sum()
sizing = np.linspace(1,number_of_splits,number_of_splits)/total_folds
split_score_columns = ['split0_test_score',
       'split1_test_score', 'split2_test_score', 'split3_test_score',
       'split4_test_score']

scores_df_rf['Updated_rank_test_score'] = scores_df_rf[split_score_columns].mul(sizing, axis=1).sum(axis=1)
scores_df_rf = scores_df_rf.sort_values(by=['Updated_rank_test_score'],ascending=False).reset_index(drop='index')
#%%
pm = scores_df_rf['params'][0]

### Random Forest with Randomized Cross Validation ####
rf = RandomForestClassifier(**pm, n_jobs=-1)
rf.fit(train_features, train_labels)

#save the model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/ML Models/')

Data_Title = 'RF Model'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(rf, open(filename, 'wb'))

roc_auc_score(train_labels, rf.predict(train_features))
confusion_matrix(train_labels, rf.predict(train_features))
accuracy_score(train_labels, rf.predict(train_features))
recall_score(train_labels, rf.predict(train_features))
precision_score(train_labels, rf.predict(train_features))

roc_auc_score(validation_labels, rf.predict(validation_features))
confusion_matrix(validation_labels, rf.predict(validation_features))
accuracy_score(validation_labels, rf.predict(validation_features))
recall_score(validation_labels, rf.predict(validation_features))
precision_score(validation_labels, rf.predict(validation_features))

print(classification_report(validation_labels, rf.predict(validation_features)))

#Creating Feature Importance Chart
feature_importance_df = pd.DataFrame(np.column_stack((feature_list,np.round(rf.feature_importances_,4))))
feature_importance_df.columns = ['Features', 'Random Forest Importance']
feature_importance_df = feature_importance_df.sort_values(by=['Random Forest Importance'], ascending=False)
feature_importance_df = feature_importance_df.set_index('Features')
feature_importance_df['Random Forest Importance'] = feature_importance_df['Random Forest Importance'].astype(float)
feature_importance_df.plot.bar()

#%%
######Randomized CV for Gradient Boosting using Timseries split #######
   
# loss type
loss = ['deviance', 'exponential']
# Learning Rate
learning_rate = np.linspace(.001, .1, num=50)
# Number of boosting stages to perform
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Sub_Sample
subsample = np.linspace(.01, 1, num=100)
# Maximum number of levels in tree
min_samples_split = [int(x) for x in np.linspace(1, 10, num = 10)]
# Max depth of individual regressors
max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
# max features
max_features = ['auto', 'sqrt', 'log2']
# tolerance for early stopping
tol = np.linspace(1e-4, .01, num=100)
# Create the random grid
random_grid_gb = {'loss': loss,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'subsample': subsample,
                'min_samples_split': min_samples_split,
                'max_depth': max_depth,
                'max_features': max_features,
                'tol': tol}
print(random_grid_gb) 

# Use the random grid to search for best hyperparameters
# First create the base model to tune
GB = GradientBoostingClassifier()
# Random search of parameters, using time series split, 
# search across 100 different combinations
GB_random = RandomizedSearchCV(estimator = GB, param_distributions = random_grid_gb, \
                                n_iter = 100, cv = tscv, verbose=2, random_state=42, \
                                    scoring='f1')
# Fit the random search model
search_GB = GB_random.fit(train_features, train_labels)

search_GB.best_params_
search_GB.cv_results_
search_GB.best_score_
search_GB.best_estimator_
search_GB.scorer_

# save the CV model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')

Data_Title = 'GB CV Results'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(search_GB, open(filename, 'wb'))
#%%
#### Load the GB Random CV results if needed #####

path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')
Data_Title = 'GB CV Results'
filename = path_to_file + Data_Title + '.sav'
search_GB = pickle.load(open(filename, 'rb'))

scores_df_gb = pd.DataFrame(search_GB.cv_results_)
total_folds = np.linspace(1,number_of_splits,number_of_splits).sum()
sizing = np.linspace(1,number_of_splits,number_of_splits)/total_folds

split_score_columns = ['split0_test_score',
       'split1_test_score', 'split2_test_score', 'split3_test_score',
       'split4_test_score']

scores_df_gb['Updated_rank_test_score'] = scores_df_gb[split_score_columns].mul(sizing, axis=1).sum(axis=1)
scores_df_gb = scores_df_gb.sort_values(by=['Updated_rank_test_score'],ascending=False).reset_index(drop='index')
#%%

pm = scores_df_gb['params'][0]
### Gradient Boosting with Randomized Cross Validation ####

GB = GradientBoostingClassifier(**pm)
GB.fit(train_features, train_labels)

#save the model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/ML Models/')

Data_Title = 'GB Model'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(GB, open(filename, 'wb'))

roc_auc_score(train_labels, GB.predict(train_features))
confusion_matrix(train_labels, GB.predict(train_features))
accuracy_score(train_labels, GB.predict(train_features))
recall_score(train_labels, GB.predict(train_features))
precision_score(train_labels, GB.predict(train_features))

roc_auc_score(validation_labels, GB.predict(validation_features))
confusion_matrix(validation_labels, GB.predict(validation_features))
accuracy_score(validation_labels, GB.predict(validation_features))
recall_score(validation_labels, GB.predict(validation_features))
precision_score(validation_labels, GB.predict(validation_features))

print(classification_report(validation_labels, GB.predict(validation_features)))

Creating Feature Importance Chart
feature_importance_df_2 = pd.DataFrame(np.column_stack((feature_list,np.round(GB.feature_importances_,4))))
feature_importance_df_2.columns = ['Features', 'Gradient Boosting Importance']
feature_importance_df_2 = feature_importance_df_2.set_index('Features')
feature_importance_df_2['Gradient Boosting Importance'] = feature_importance_df_2['Gradient Boosting Importance'].astype(float)
FEATURE_DF = pd.concat([feature_importance_df, feature_importance_df_2], axis = 1, ignore_index=False)

SMALL_SIZE = 5
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
FEATURE_DF.plot.bar(title='Feature Importance')

ax = FEATURE_DF.plot.bar(title='Feature Importance')

fig = ax.get_figure()
Chart = 'Feature Importance'
path_to_file = (
    r'/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On/Sentiment Trading/Visuals/'
        )
output_name = path_to_file + Chart + '.pdf'
fig.savefig(output_name)
#%%
######Randomized CV for K nearest neighbors using Timseries split #######

# First create the base model to tune
neigh = Pipeline([('scaler', StandardScaler()), ('knearest', KNeighborsClassifier())])    
neigh.get_params().keys()
   
# Number of neighbors
n_neighbors = [int(x) for x in np.linspace(start = 3, stop = 20, num = 2)]
# Weight function used in prediction
weights = ['uniform', 'distance']
# Algorithm used to compute the nearest neighbors
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
# Leaf_size
leaf_size = [int(x) for x in np.linspace(start = 10, stop = 50, num = 2)]

random_grid = {'knearest__n_neighbors': n_neighbors,
                'knearest__weights': weights,
                'knearest__algorithm': algorithm,
                'knearest__leaf_size': leaf_size}
print(random_grid) 

# Random search of parameters, using time series split, 
# search across 100 different combinations
neigh_random = RandomizedSearchCV(estimator = neigh, param_distributions = random_grid, \
                                n_iter = 100, cv = tscv, verbose=2, random_state=42, \
                                    scoring='f1')
# Fit the random search model
search_neigh = neigh_random.fit(train_features, train_labels)

search_neigh.best_params_
search_neigh.cv_results_
search_neigh.best_score_
search_neigh.best_estimator_
search_neigh.scorer_

# save the CV model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')

Data_Title = 'KNNeighbors CV Results'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(search_neigh, open(filename, 'wb'))
#%%
### Load the RF Random CV results if needed #####

path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')
Data_Title = 'KNNeighbors CV Results'
filename = path_to_file + Data_Title + '.sav'
search_neigh = pickle.load(open(filename, 'rb'))
#%%
Neigh = search_neigh.best_estimator_

Neigh.fit(train_features, train_labels)

#save the model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/ML Models/')

Data_Title = 'Neigh Model'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(Neigh, open(filename, 'wb'))

roc_auc_score(train_labels, Neigh.predict(train_features))
confusion_matrix(train_labels, Neigh.predict(train_features))
accuracy_score(train_labels, Neigh.predict(train_features))
recall_score(train_labels, Neigh.predict(train_features))
precision_score(train_labels, Neigh.predict(train_features))

roc_auc_score(validation_labels, Neigh.predict(validation_features))
confusion_matrix(validation_labels, Neigh.predict(validation_features))
accuracy_score(validation_labels, Neigh.predict(validation_features))
recall_score(validation_labels, Neigh.predict(validation_features))
precision_score(validation_labels, Neigh.predict(validation_features))

print(classification_report(validation_labels, Neigh.predict(validation_features)))
#%%
######Randomized CV for Ada Boost using Timseries split #######

# First create the base model to tune
AdaBoost = Pipeline([('scaler', StandardScaler()), ('AdaBoost', AdaBoostClassifier())])    
AdaBoost.get_params().keys()
   
# Number of estimators
n_estimators = [int(x) for x in np.linspace(start = 30, stop = 200, num = 10)]

# Learning rate
learning_rate = [np.round(x,2) for x in np.linspace(start = .10, stop = 2.0, num = 20)]

# Weight function used in prediction
#algorithm = ['SAMME','SAMME.R']

random_grid = {'AdaBoost__n_estimators': n_estimators,
                'AdaBoost__learning_rate': learning_rate,
                }
print(random_grid) 

# Random search of parameters, using time series split, 
# search across 100 different combinations
adaboost_random = RandomizedSearchCV(estimator = AdaBoost, param_distributions = random_grid, \
                                n_iter = 100, cv = tscv, verbose=2, random_state=42, \
                                    scoring='f1')
# Fit the random search model
search_adaboost = adaboost_random.fit(train_features, train_labels)

search_adaboost.best_params_
search_adaboost.cv_results_
search_adaboost.best_score_
search_adaboost.best_estimator_
search_adaboost.scorer_

# save the CV model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')

Data_Title = 'AdaBoost CV Results'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(search_adaboost, open(filename, 'wb'))

#### Load the AdaBoost Random CV results if needed #####
#%%
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')
Data_Title = 'Adaboost CV Results'
filename = path_to_file + Data_Title + '.sav'
search_adaboost = pickle.load(open(filename, 'rb'))
#%%

### Adaboost with Randomized Cross Validation ####
ABoost = search_adaboost.best_estimator_
ABoost.fit(train_features, train_labels)

#save the model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/ML Models/')

Data_Title = 'ABoost Model'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(ABoost, open(filename, 'wb'))

roc_auc_score(train_labels, ABoost.predict(train_features))
confusion_matrix(train_labels, ABoost.predict(train_features))
accuracy_score(train_labels, ABoost.predict(train_features))
recall_score(train_labels, ABoost.predict(train_features))
precision_score(train_labels, ABoost.predict(train_features))

roc_auc_score(validation_labels, ABoost.predict(validation_features))
confusion_matrix(validation_labels, ABoost.predict(validation_features))
accuracy_score(validation_labels, ABoost.predict(validation_features))
recall_score(validation_labels, ABoost.predict(validation_features))
precision_score(validation_labels, ABoost.predict(validation_features))

print(classification_report(validation_labels, ABoost.predict(validation_features)))
#%%
###### Randomized Search for Gaussian Process  #####

# First create the base model to tune
GP_Classifier = Pipeline([('scaler', StandardScaler()), ('Gaussian_Process', GaussianProcessClassifier())])    

# The number of restarts of the optimizer for finding the kernel’s parameters 
#which maximize the log-marginal likelihood
n_restarts_optimizer = [int(x) for x in np.linspace(start = 0, stop = 3, num = 4)]

# The maximum number of iterations in Newton’s method for approximating the 
#posterior during predict
max_iter_predict = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]

#If warm-starts are enabled, the solution of the last Newton iteration on 
#the Laplace approximation of the posterior mode is used as initialization 
#for the next call
warm_start = ['True','False']

random_grid = {'Gaussian_Process__n_restarts_optimizer': n_restarts_optimizer,
                'Gaussian_Process__max_iter_predict': max_iter_predict,
                'Gaussian_Process__warm_start': warm_start,
                }
print(random_grid) 

# Random search of parameters, using time series split, 
# search across 100 different combinations
GP_Classifier_random = RandomizedSearchCV(estimator = GP_Classifier, param_distributions = random_grid, \
                                n_iter = 5, cv = tscv, verbose=2, random_state=42, \
                                    scoring='f1')
# Fit the random search model
search_GP = GP_Classifier_random.fit(train_features, train_labels)

search_GP.best_params_
search_GP.cv_results_
search_GP.best_score_
search_GP.best_estimator_
search_GP.scorer_

# save the CV model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')

Data_Title = 'Gaussian Process CV Results'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(search_GP, open(filename, 'wb'))
#%%

#### Load the Gassian Process Random CV results if needed #####

path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/')
Data_Title = 'Gaussian Process CV Results'
filename = path_to_file + Data_Title + '.sav'
search_GP = pickle.load(open(filename, 'rb'))
#%%
### Gaussian Process with Randomized Cross Validation ####
GP = search_GP.best_estimator_
GP.fit(train_features, train_labels)

#save the model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/Sentiment Research/ML Models/')

Data_Title = 'GP Model'
filename = path_to_file + Data_Title + '.sav'
pickle.dump(GP, open(filename, 'wb'))

roc_auc_score(train_labels, GP.predict(train_features))
confusion_matrix(train_labels, GP.predict(train_features))
accuracy_score(train_labels, GP.predict(train_features))
recall_score(train_labels, GP.predict(train_features))
precision_score(train_labels, GP.predict(train_features))

roc_auc_score(validation_labels, GP.predict(validation_features))
confusion_matrix(validation_labels, GP.predict(validation_features))
accuracy_score(validation_labels, GP.predict(validation_features))
recall_score(validation_labels, GP.predict(validation_features))
precision_score(validation_labels, GP.predict(validation_features))

print(classification_report(validation_labels, GP.predict(validation_features)))
#%%
#### Create Predictions from Ensemble of acurate algos ####
ML_Model = rf
rf_predictions_train = ML_Model.predict(train_features)
rf_predictions_val = ML_Model.predict(validation_features)

RF_PREDICTIONS = np.concatenate([rf_predictions_train, rf_predictions_val])

ML_Model = GB
gb_predictions_train = ML_Model.predict(train_features)
gb_predictions_val = ML_Model.predict(validation_features)

GB_PREDICTIONS = np.concatenate([gb_predictions_train, gb_predictions_val])

ML_Model = Neigh
neigh_predictions_train = ML_Model.predict(train_features)
neigh_predictions_val = ML_Model.predict(validation_features)

NEIGH_PREDICTIONS = np.concatenate([neigh_predictions_train, neigh_predictions_val])

ML_Model = ABoost
adaboost_predictions_train = ML_Model.predict(train_features)
adaboost_predictions_val = ML_Model.predict(validation_features)

ADBOOST_PREDICTIONS = np.concatenate([adaboost_predictions_train, adaboost_predictions_val])

ML_Model = GP
gp_predictions_train = ML_Model.predict(train_features)
gp_predictions_val = ML_Model.predict(validation_features)

GP_PREDICTIONS = np.concatenate([gp_predictions_train, gp_predictions_val])

PREDICTIONS = pd.DataFrame(np.column_stack((RF_PREDICTIONS, GB_PREDICTIONS, \
                                            NEIGH_PREDICTIONS, ADBOOST_PREDICTIONS, GP_PREDICTIONS)))
PREDICTIONS.columns = ['Random Forest', 'Gradient Boosting', 'K Nearest Neighbors', 'Adaboost', 'Gaussian Process']
#%%
import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
# Compute the correlation matrix
corr = PREDICTIONS.corr() #see correlation between model predictions
corr
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 7))
ax.set_title('Model Prediction Correlation')
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.3, cbar_kws={"shrink": .8}, annot=True)

fig = ax.get_figure()
Chart = 'Model Prediction Correlation'
path_to_file = (
    r'/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On/Sentiment Trading/Visuals/'
        )
output_name = path_to_file + Chart + '.pdf'
fig.savefig(output_name)
#%%

#At least 3 out of 5 algos voting positive, otherwise go with 0
PREDICTIONS['Vote'] = np.where(PREDICTIONS.mean(axis=1)>.51,1,0)

PREDICTIONS_val = pd.DataFrame(np.column_stack\
                               ((rf_predictions_val, gb_predictions_val,\
                                 neigh_predictions_val, adaboost_predictions_val, gp_predictions_val)))
PREDICTIONS_val['Vote'] = np.round(PREDICTIONS_val.mean(axis=1),0)

accuracy_score(validation_labels, np.array(PREDICTIONS_val['Vote']))
confusion_matrix(validation_labels, np.array(PREDICTIONS_val['Vote']))
f1_score(validation_labels, np.array(PREDICTIONS_val['Vote']))
roc_auc_score(validation_labels, np.array(PREDICTIONS_val['Vote']))
print(classification_report(validation_labels, np.array(PREDICTIONS_val['Vote'])))
#%%
features['Prediction'] = PREDICTIONS['Vote'].values

#########  Portfolio Construction  ########
Predictions_DF = features.iloc[:,(-len(tickers)-1):]

#Using the same in sample dates here and for equal weight benchmark
f_date = datetime.datetime.strptime(In_Sample_Start, '%Y-%m-%d')
l_date = datetime.datetime.strptime(In_Sample_End,  '%Y-%m-%d')

delta = l_date - f_date

#Choose the number of periods (days in range / the forecasted return days)
period_delta = np.floor(delta.days/(Number_days_to_hold))
period_delta = int(period_delta)
first_period = In_Sample_Start #using f_date

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

print("After market impact, you would deduct " + \
      str(np.round((Total_Trades * slippage * 1/number_tickers)*100,2)) + \
          " % from gross performance and " + str(Total_Trades*trade_commission) + " dollars.")

####   Equal Weight Portfolio ###

Equal_Weight_Returns =  Sector_Adj_Close_Prices[In_Sample_Start:In_Sample_End]
Equal_Weight_Returns = Equal_Weight_Returns.pct_change().dropna()
Equal_Weight_Returns = Equal_Weight_Returns.div(len(tickers))
Equal_Weight_Returns['Equal Weight Returns'] = Equal_Weight_Returns.sum(axis=1)
Equal_Weight_Returns['Equal Weight Index'] = (1 + Equal_Weight_Returns['Equal Weight Returns']).cumprod()
#%%
portfolio_index = (1 + returns_df_portfolio_2).cumprod()
portfolio_index.columns = ['Sentiment Strategy']
plt.rcParams.update(plt.rcParamsDefault)
ax = portfolio_index.plot(title='Sentiment In Sample performance', logy=True)
ax.axvline(pd.to_datetime('2011-12-22'), color='r', linestyle='--', lw=1)
fig = ax.get_figure()
Performance_Chart = 'Sentiment In Sample Performance Chart'
path_to_file = (
    r'/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On/Sentiment Trading/Visuals/'
        )
output_name = path_to_file + Performance_Chart + '.pdf'
fig.savefig(output_name)
#%%
##### Slice off just Validation set ####
rowlength = returns_df_portfolio_2.shape[0]
portfolio_returns_val = returns_df_portfolio_2.iloc[int(-rowlength*test_ratio):,:]
portfolio_returns_val.head()
portfolio_index_val = (1 + portfolio_returns_val).cumprod()
portfolio_index_val.columns = ['Sentiment Strategy']

ax = portfolio_index_val.plot(title='Sentiment In Sample Validation performance')
fig = ax.get_figure()
Performance_Chart = 'Sentiment In Sample Validation Performance Chart'
path_to_file = (
    r'/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On/Sentiment Trading/Visuals/'
        )
output_name = path_to_file + Performance_Chart + '.pdf'
fig.savefig(output_name)
#%%
##### Performance Stats ####

results_performance = pd.DataFrame()
D = len(portfolio_index_val)
perf = ((portfolio_index_val.iloc[-1,-1])**((252/Number_days_to_hold)/D))-1
std = portfolio_index_val.pct_change().std() * (252/Number_days_to_hold) ** 0.5
Sharpe = perf/std
roll_max = portfolio_index_val.expanding().max()
daily_drawdown = portfolio_index_val/roll_max - 1.0
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


Validation_Start_Date = portfolio_index_val.first_valid_index() + pd.tseries.offsets.BusinessDay(-1)
Equal_Weight_Returns_Validation = Equal_Weight_Returns['Equal Weight Returns'][Validation_Start_Date:]
Equal_Weight_Returns_Validation.head()

EW_Returns_Validation = Equal_Weight_Returns_Validation
EW_Returns_Validation_Reset_Index = (1+EW_Returns_Validation).cumprod()
results_performance_eq = pd.DataFrame()
D = len(EW_Returns_Validation_Reset_Index)
perf = EW_Returns_Validation_Reset_Index.iloc[-1,]**(252/D)-1
std = EW_Returns_Validation_Reset_Index.pct_change().std() * (252) ** 0.5
Sharpe = perf/std
roll_max = EW_Returns_Validation_Reset_Index.expanding().max()
daily_drawdown = EW_Returns_Validation_Reset_Index/roll_max - 1.0
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

Portfolio_Performance = 'Validation Portfolio Performance'

path_to_file = (r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/'
                )
output_name = path_to_file + Portfolio_Performance + '.csv'
performance_combo.to_csv(output_name)

#merge the strategy and equal weight indices
performance_results = pd.DataFrame()
performance_results = pd.concat([performance_results, portfolio_index_val], axis = 1, ignore_index=False)
performance_results = pd.concat([performance_results, EW_Returns_Validation_Reset_Index], axis = 1, ignore_index=False)
performance_results = performance_results.fillna(method='ffill').fillna(value=1.0)
performance_results.iloc[0,0:2] = 1 #set initial value as 1
performance_results.tail()

ax = performance_results.plot(title='In Sample Validation performance')
fig = ax.get_figure()
Performance_Chart = 'In Sample Validation Performance Chart'
path_to_file = (
    r'/Users/downey/Dropbox/Holborn Assets/Research Papers I Am Working On/Sentiment Trading/Visuals/'
        )
output_name = path_to_file + Performance_Chart + '.pdf'
fig.savefig(output_name)
#%%



