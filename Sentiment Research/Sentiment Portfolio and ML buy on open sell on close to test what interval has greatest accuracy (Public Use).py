#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:13:17 2020

@author: downey
"""
import os
import datetime
import time
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

##############################################################################

#### Sentiment and Machine Learning for Portfolio ###

##############################################################################

pd.set_option('display.max_columns', 200)
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
    pd.read_csv('price data')
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

#Didnt' find any value add for SAHM_Rule

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
Volume = pd.concat([Volume, Share_Traded], axis=1, ignore_index=False)
Volume = pd.concat([Volume, Total_shares_sum], axis=1, ignore_index=False)

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
####Downloading SPDR Sector ETFs ####

Sector_tickers = ['XLE','XLY','XLP','XLF','XLV','XLI','XLB','XLK','XLU']

data = yf.download(Sector_tickers, start="1999-01-01", end="2020-06-30")

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
#%%

Models = [['SVC'], 
        ['K Nearest Neighbors'],
        ['AdaBoost'],
        ['MLP Neural Network'],
        ['Gaussian Process'],
        ['Gradient Boosting'],
        ['Random Forest']] 
  
# Create the pandas DataFrame 
Models_Accuracy_df = pd.DataFrame(Models, columns = ['Model'])

# Create the pandas DataFrame 
Models_f1_df = pd.DataFrame(Models, columns = ['Model'])
#%%
t0 = time.time()
for DAYS in range(1,30,1):
    #create for loop of buy on open and sell on t+n close
    Number_days_to_hold = DAYS
    
    Returns_Open_Close_3_days = pd.DataFrame()
    Returns_Open_Close_3_days = pd.DataFrame(np.nan, index=adjusted_df.index, columns=['A', 'B','C','D','E','F','G','H','I'])
    
    for j in range(0, len(Sector_tickers)):
        for i in range(0, len(adjusted_df)-2):
            Returns_Open_Close_3_days.iloc[i, j] = (adjusted_df.iloc[i+2,j+len(Sector_tickers)]/adjusted_df.iloc[i,j])-1
    
    Returns_Open_Close_3_days.columns = Sector_tickers
    Sector_Returns_forward_n_days = Returns_Open_Close_3_days.dropna()
    Sector_Returns_forward_n_days.index = pd.to_datetime(Sector_Returns_forward_n_days.index)
    Sector_Returns_forward_n_days.columns = [str(col) + '_' + str(Number_days_to_hold)+'day_return' for col in Sector_Returns_forward_n_days.columns]
    
    ##### Create Data for Each Ticker ####
    
    #####
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
    results_9.columns
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
    sectors.index
    
    DF12 = pd.concat([DF11,sectors], axis = 1, ignore_index=False)
    
    DF12 = DF12.drop(columns=['variable'])
    
    #keep only every nth row so as to have IID samples
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
    
    # Split the data into training and testing sets, with simple split
    train_features, validation_features, train_labels, validation_labels = \
        train_test_split(features_array, labels, test_size=0.20, random_state=42, \
                         shuffle=False)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', validation_features.shape)
    print('Testing Labels Shape:', validation_labels.shape)
    
    number_features = train_features.shape[1]
    
    #### SVC Linear #####
    
    SVC_model = make_pipeline(StandardScaler(), \
                              LinearSVC(random_state=42, dual=False, tol=1e-5,\
                                        verbose=1,
                                        class_weight='balanced'))
    
    SVC_model.fit(train_features, train_labels)
    
    SVC_Accuracy = accuracy_score(validation_labels, SVC_model.predict(validation_features))
    SVC_f1 = f1_score(validation_labels, SVC_model.predict(validation_features))

    #### K Nearest Neighbors#####
    
    neigh = make_pipeline(StandardScaler(), \
                              KNeighborsClassifier())
    neigh.fit(train_features, train_labels)
    
    K_nearest_Accuracy = accuracy_score(validation_labels, neigh.predict(validation_features))
    K_nearest_f1 = f1_score(validation_labels, neigh.predict(validation_features))

    #### AdaBoost #####
    
    AdaBoost_clf = make_pipeline(StandardScaler(), \
                              AdaBoostClassifier(random_state=42))
    AdaBoost_clf.fit(train_features, train_labels)
    
    AdaBoost_Accuracy = accuracy_score(validation_labels, AdaBoost_clf.predict(validation_features))
    AdaBoost_f1 = f1_score(validation_labels, AdaBoost_clf.predict(validation_features))

    #### MLP Neural Network #####
    
    MLP_clf = make_pipeline(StandardScaler(), \
                              MLPClassifier(verbose=1, random_state=42))
    MLP_clf.fit(train_features, train_labels)
    
    MLP_Accuracy = accuracy_score(validation_labels, MLP_clf.predict(validation_features))
    MLP_f1 = f1_score(validation_labels, MLP_clf.predict(validation_features))

    ###### GaussianProcess #####
    
    GP_clf = make_pipeline(StandardScaler(), \
                              GaussianProcessClassifier(random_state=42))
    GP_clf.fit(train_features, train_labels)
    
    Gaussian_Accuracy = accuracy_score(validation_labels, GP_clf.predict(validation_features))
    Gaussian_f1 = f1_score(validation_labels, GP_clf.predict(validation_features))

    ####Gradient Boosting#####
    
    GB = GradientBoostingClassifier(verbose=1, random_state=42)
    GB.fit(train_features, train_labels)
    
    Gradient_Boosting_Accuracy = accuracy_score(validation_labels, GB.predict(validation_features))
    Gradient_Boosting_f1 = f1_score(validation_labels, GB.predict(validation_features))

    ### Random Forest ####
    
    rf = RandomForestClassifier(n_jobs=-2, verbose=1, random_state=42)
    rf.fit(train_features, train_labels)
    
    RandomForest_Accuracy = accuracy_score(validation_labels, rf.predict(validation_features))
    RandomForest_f1 = f1_score(validation_labels, rf.predict(validation_features))

    #### Combining Accuracy Scores ####
    
    data_accuracy_scores = [[SVC_Accuracy], 
            [K_nearest_Accuracy],
            [AdaBoost_Accuracy],
            [MLP_Accuracy],
            [Gaussian_Accuracy],
            [Gradient_Boosting_Accuracy],
            [RandomForest_Accuracy]] 
    
    data_f1_scores = [[SVC_f1], 
            [K_nearest_f1],
            [AdaBoost_f1],
            [MLP_f1],
            [Gaussian_f1],
            [Gradient_Boosting_f1],
            [RandomForest_f1]] 
      
    # Create the pandas DataFrame 
    df_accuracy_scores = pd.DataFrame(data_accuracy_scores, columns = [str(Number_days_to_hold)])
    Models_Accuracy_df= pd.concat([Models_Accuracy_df,df_accuracy_scores], axis = 1, ignore_index=False)

    df_f1_scores = pd.DataFrame(data_f1_scores, columns = [str(Number_days_to_hold)])
    Models_f1_df= pd.concat([Models_f1_df,df_f1_scores], axis = 1, ignore_index=False)

t1 = time.time()
print("It took " + str(np.round((t1-t0)/60,2)) + " minutes.")
#%%
Models_Accuracy_df    
    
Models_Accuracy_df = Models_Accuracy_df.set_index('Model')
Models_Accuracy_df.loc['Average'] = Models_Accuracy_df.mean()
Models_Accuracy_df.loc['Std. Dev.'] = Models_Accuracy_df.iloc[:-1,:].std()
Models_Accuracy_df['Model Average'] = Models_Accuracy_df.mean(axis=1)
print(Models_Accuracy_df)

Models_f1_df = Models_f1_df.set_index('Model')
Models_f1_df.loc['Average'] = Models_f1_df.mean()
Models_f1_df.loc['Std. Dev.'] = Models_f1_df.iloc[:-1,:].std()
Models_f1_df['Model Average'] = Models_f1_df.mean(axis=1)
print(Models_f1_df)
#%%

#Save the performance for later use

Data_Title = 'Model Accuracies Over Time'

path_to_file = (r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/'
                )
output_name = path_to_file + Data_Title + '.csv'
Models_Accuracy_df.to_csv(output_name)

#Save the performance for later use

Data_Title = 'Model f1 Over Time'

path_to_file = (r'/Users/downey/coding/Python/Scripts/Sentiment Research/Analysis/'
                )
output_name = path_to_file + Data_Title + '.csv'
Models_f1_df.to_csv(output_name)
#%%

