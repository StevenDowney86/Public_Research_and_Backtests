#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:05:37 2019

@author: downey
"""
import time
import datetime
import pickle
import os

import pandas as pd
import quandl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas_datareader as pdr

pd.set_option('display.max_columns', 110)
pd.set_option('display.max_rows', 1000)
quandl.ApiConfig.api_key = "apikey"
#turn off pandas warning for index/slicing as copy warning
pd.options.mode.chained_assignment = None  # default='warn'

os.chdir(
    '/Users/downey/coding/Python/Scripts/EPAT Project RF Models,'
    ' Code, and Results'
    )

##############################################################################

#Random Forests to Predict 12 months ahead Price to Free Cash Flow

##############################################################################
#%%

#############Creating the Macroeconomic feature set#########################

start = datetime.datetime(1996, 1, 1)
start_quandl = "1996-01-01" #create date to be used with Quandl function
end = datetime.datetime(2019, 9, 30)
end_quandl = "2019-09-30"

CPI = pdr.get_data_fred('CPIAUCSL', start, end)
CPI.columns = ['Headline CPI']
CPI['CPI YoY Change'] = CPI.pct_change(12)
#create 1 month lag since we won't know CPI data until about 1 month after data point
CPI['CPI YoY Change Lag'] = CPI['CPI YoY Change'].shift(-1)

Core_CPI = pdr.get_data_fred('CPILFESL', start, end)
Core_CPI.columns = ['Core CPI']
Core_CPI['Core CPI YoY Change'] = Core_CPI.pct_change(12)
#create 1 month lag since we won't know CPI data until about 1 month after data point
Core_CPI['Core CPI YoY Change Lag'] = Core_CPI['Core CPI YoY Change'].shift(-1)

WTI_Oil = pdr.get_data_fred('DCOILWTICO', start, end)
WTI_Oil.columns = ['WTI Oil']

Gold = pdr.get_data_fred('GOLDAMGBD228NLBM', start, end)
Gold.columns = ['Gold']

Treasury_Yields = pdr.get_data_fred(['DTB3', 'DGS10'], start, end)
Treasury_Yields['10yr_3mo_spread'] = Treasury_Yields['DGS10']-Treasury_Yields['DTB3']
Treasury_Yields.columns = ['3 Month Treasury Yield', '10 Year Treasury Yield', '10yr-3mo Yield']

LIBOR_3mo_USD = pdr.get_data_fred('USD3MTD156N', start, end)
LIBOR_3mo_USD.columns = ['LIBOR 3mo USD']

#create spread between OAS of high yield and A rated
Credit_Spreads = pdr.get_data_fred(['BAMLH0A0HYM2', 'BAMLC0A3CA'], start, end)
Credit_Spreads['spread'] = Credit_Spreads['BAMLH0A0HYM2']-Credit_Spreads['BAMLC0A3CA']
Credit_Spreads.columns = ['High Yield OAS', 'Inv Grade A OAS', 'Credit Spread']

Dollar_Index = pdr.get_data_fred('DTWEXB', start, end)
Dollar_Index.columns = ['Dollar Index']

Unemployment_Rate = pdr.get_data_fred('UNRATE', start, end)
Unemployment_Rate.columns = ['Unemployment Rate (U-3)']
#create 1 month lag since we won't know CPI data until about 1 month after data point
Unemployment_Rate['Unemployment Rate (U-3) Lag'] = \
    Unemployment_Rate['Unemployment Rate (U-3)'].shift(-1)

Real_Personal_Consumption_Expenditures = pdr.get_data_fred('DPCERAM1M225NBEA', start, end)
Real_Personal_Consumption_Expenditures.columns = ['Real_Personal_Consumption_Expenditures']
#convert to percent decimal
Real_Personal_Consumption_Expenditures = Real_Personal_Consumption_Expenditures/100
#create 1 month lag since we won't know data about 1 month after data time point
Real_Personal_Consumption_Expenditures['Real_Personal_Consumption_Expenditures Lag'] \
    = Real_Personal_Consumption_Expenditures['Real_Personal_Consumption_Expenditures'].shift(-1)

Building_Permits = pdr.get_data_fred('PERMIT', start, end)
Building_Permits.columns = ['Building permits, new private housing units']
#create 1 month lag since we won't know data about 1 month after data time point
Building_Permits['Building permits, new private housing units Lag'] = \
    Building_Permits['Building permits, new private housing units'].shift(-1)

Wilshire_5000_Index = pdr.get_data_fred('WILL5000IND', start, end)
Wilshire_5000_Index.columns = ['Wilshire 5000 TR Index']

ISM_Manufacturing_PMI = quandl.get("ISM/MAN_PMI", start_date=start_quandl, end_date=end_quandl)
ISM_Manufacturing_PMI.columns = ['ISM Manufacturing PMI Index']

TED_Rate = pdr.get_data_fred('TEDRATE', start, end)
TED_Rate.columns = ['TED Rate Spread']

results = pd.DataFrame()
results = pd.concat([results, CPI['CPI YoY Change Lag']], axis = 1, ignore_index = False)
results = pd.concat([results, Core_CPI['Core CPI YoY Change Lag']], axis = 1, ignore_index = False)
results = pd.concat([results, WTI_Oil], axis = 1, ignore_index = False)
results = pd.concat([results, Gold], axis = 1, ignore_index = False)
results = pd.concat([results, Treasury_Yields], axis = 1, ignore_index = False)
results = pd.concat([results, LIBOR_3mo_USD], axis = 1, ignore_index = False)
results = pd.concat([results, Credit_Spreads], axis = 1, ignore_index = False)
results = pd.concat([results, Dollar_Index], axis = 1, ignore_index = False)
results = pd.concat([results, Unemployment_Rate['Unemployment Rate (U-3) Lag']], \
                    axis = 1, ignore_index = False)
results = pd.concat([results, Real_Personal_Consumption_Expenditures\
                     ['Real_Personal_Consumption_Expenditures Lag']], \
                    axis = 1, ignore_index = False)
results = pd.concat([results, Building_Permits\
                     ['Building permits, new private housing units Lag']]\
                    , axis = 1, ignore_index = False)
results = pd.concat([results, Wilshire_5000_Index], axis = 1, ignore_index = False)
results = pd.concat([results, ISM_Manufacturing_PMI], axis = 1, ignore_index = False)
results = pd.concat([results, TED_Rate], axis = 1, ignore_index = False)

#need to change index to dattime in order to resample below
index_data = pd.to_datetime(results.index)
results = results.set_index(index_data)

#we will need to convert to quarterly data to eventually merge with fundamental
#feature data
Macro_Econ_Features_quarterly = results.resample('Q').mean()
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

###########################Choosing the Sector to Test########################

#filter out the tickers based on specified sector, industry, or highest market cap

tickers_df1.sector.unique()

tickers_df1.industry.unique()

tickers_df1.scalemarketcap.unique()
#%%
t0 = time.time()
test_sector = 'Healthcare' #you will need to cycle through each sector

sector_stocks = tickers_df1[tickers_df1['sector'] == test_sector]

#put tickers to list from sector specified
sector_tickers = sector_stocks['ticker'].tolist()
len(sector_tickers)
#fundamentals imported already
fundamentals = USA_fundamentals[USA_fundamentals.ticker.isin(sector_tickers)]

#Choose dimension rolling twelve month as reported 'ART'
fundamentals = fundamentals[fundamentals.dimension == 'ART']

###########################Creating Training/Validation Set###################

# Converting to date index which will be needed when doing the ML training
fundamentals['calendardate'] = pd.to_datetime(fundamentals['calendardate'])

#create test/validation dataset

start_date = '01-01-1999'
end_date = '12-31-2012'

#create mask to be used to filter out dataframe based on dates
mask = (fundamentals['calendardate'] > start_date) & (fundamentals['calendardate'] <= end_date)

fundamentals = fundamentals.loc[mask]

#Find data rows where fundamentals have been restated for previous quarter
#and we want to remove for backtesting since at the time you only have the first
#release of data and not subsequent revisions
duplicateRowsDF = fundamentals[fundamentals.duplicated(['ticker', 'calendardate'])]

print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')

fundamentals = fundamentals.drop_duplicates(subset = ['ticker', 'calendardate'],\
                                            keep = 'first')

duplicateRowsDF = fundamentals[fundamentals.duplicated(['ticker', 'calendardate'])]

print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')

#######################Create Fundamental Feature Set#########################

#######Manufactured Features#################################################

#Net Operating Profit After Taxes
fundamentals['NOPAT'] = fundamentals['ebit'] - fundamentals['taxexp']

#capex to sales ratio
fundamentals['capex_to_sales'] = fundamentals['capex']/fundamentals['revenue']

#YoY Revenue Growth by Ticker
fundamentals['YoY_Rev_Growth'] = fundamentals.groupby('ticker')['revenue'].pct_change(4)

#YoY Earnings Growth by Ticker
fundamentals['YoY_NetInc_Growth'] = fundamentals.groupby('ticker')['netinc'].pct_change(4)

#YoY Cash Flow From Operations
fundamentals['YoY_NCFO_Growth'] = fundamentals.groupby('ticker')['ncfo'].pct_change(4)

####Creating Base Case Net Operating Profit After Tax Secenario for Sector#####

#asertain the base case stats for the industry that would indicate individual
#firm reversion to the mean, using concepts from Damdoran growth rates. Will try
#1 year median industry growth rate

fundamentals['Company_YoY_NetInc_Growth'] = fundamentals.groupby('ticker')['netinc'].pct_change(4)

# start_date = '01-01-1999' end_date = '12-31-2012'
dates = pd.date_range(start='1/1/1999', periods=14*4, freq='Q')
dates2 = dates.strftime('%Y-%m-%d')
dates2 = list(dates2)

dicts = {}
results = []
for i in dates2:
    data1 = fundamentals[fundamentals['calendardate'] == i]
    median = data1['Company_YoY_NetInc_Growth'].median()
    results.append(median)

#change the column type to a string for mapping dictionary
fundamentals = fundamentals.astype({"calendardate": str})

#create the dictionary with values and keys as dates
keys = dates2
values = results
Dictionary_Industry_values = dict(zip(keys, values))

#map the values of the dictionary onto the calendardate keys in the dataset
fundamentals['Sector YoY Earnings Average'] = fundamentals\
    ['calendardate'].map(Dictionary_Industry_values)

fundamentals['Sector_YoY_Earnings_Average'] = fundamentals\
    ['Sector YoY Earnings Average']

fundamentals['Ratio_Comp_YoY_Earn_Rate_vs_Sector'] = fundamentals\
    ['Company_YoY_NetInc_Growth']/ fundamentals['Sector YoY Earnings Average']
fundamentals['Ratio_Comp_YoY_Earn_Rate_vs_Sector'] = fundamentals\
    ['Ratio_Comp_YoY_Earn_Rate_vs_Sector']

####    Creating Sales Growth Base Case    ####
fundamentals['Company_YoY_Revenue_Growth'] = fundamentals.groupby('ticker')\
    ['revenue'].pct_change(4)
fundamentals = fundamentals.dropna()

# start_date = '01-01-1999' end_date = '12-31-2012'
dates = pd.date_range(start='1/1/1999', periods=14*4, freq='Q')
dates2 = dates.strftime('%Y-%m-%d')
dates2 = list(dates2)

dicts = {}
results = []
for i in dates2:
    data1 = fundamentals[fundamentals['calendardate'] == i]
    mean = data1['Company_YoY_Revenue_Growth'].median()
    results.append(mean)

#create the dictionary with values and keys as dates
keys = dates2
values = results
Dictionary_Industry_values = dict(zip(keys, values))

#map the values of the dictionary onto the calendardate keys in the dataset
fundamentals['Sector YoY Revenue Average'] = fundamentals\
    ['calendardate'].map(Dictionary_Industry_values)
fundamentals['Sector_YoY_Revenue_Average'] = fundamentals\
    ['Sector YoY Revenue Average']

fundamentals['Ratio_Comp_YoY_Reve_Rate_vs_Sector'] = fundamentals\
    ['Company_YoY_Revenue_Growth']/ fundamentals['Sector YoY Revenue Average']
fundamentals['Ratio_Comp_YoY_Reve_Rate_vs_Sector'] = fundamentals\
    ['Ratio_Comp_YoY_Reve_Rate_vs_Sector']

####    Earnings Accrual    ####
"""
https://www.oldschoolvalue.com/valuation-methods/you-need-to-determine-earnings-quality-through-accruals/
"""

#Net Operating Assets = (Total Assets - Cash) - (Total Liabilities - Total Debt)
fundamentals['NOA'] = (fundamentals['assets']-fundamentals['cashneq']) - \
    (fundamentals['liabilities']-fundamentals['debt'])

#create BS_Aggreate_Accruals_Numerator
fundamentals['Balance_Sheet_Aggregrate_Accruals_num'] = fundamentals.groupby\
    ('ticker')['NOA'].diff(1)

#create BS_Aggregrate_Accruals_Denominator (mean of current and last period NOA)
fundamentals['Balance_Sheet_Aggregrate_Accruals_den'] = \
    fundamentals.groupby('ticker')['NOA'].apply(lambda x: x.rolling\
                                                (center=False, window=2).sum())

fundamentals['Balance_Sheet_Accruals_Ratio'] = \
    fundamentals['Balance_Sheet_Aggregrate_Accruals_num'] / \
        (fundamentals['Balance_Sheet_Aggregrate_Accruals_den']/2)

##############################################################################

#create a 4 quarter lag as 4 quarters ago fundamentals are all we have to forecast
#next full years earnings

lag_number = -4

fundamentals['fcf_lag'] = fundamentals['fcf'].shift(lag_number)
fundamentals['Revenue_lag'] = fundamentals['revenue'].shift(lag_number)
fundamentals['capex_lag'] = fundamentals['capex'].shift(lag_number)
fundamentals['de_lag'] = fundamentals['de'].shift(lag_number)
fundamentals['ebitda_lag'] = fundamentals['ebitda'].shift(lag_number)
fundamentals['opinc_lag'] = fundamentals['opinc'].shift(lag_number)
fundamentals['taxexp_lag'] = fundamentals['taxexp'].shift(lag_number)
fundamentals['gp_lag'] = fundamentals['gp'].shift(lag_number)
fundamentals['rnd_lag'] = fundamentals['rnd'].shift(lag_number)
fundamentals['sgna_lag'] = fundamentals['sgna'].shift(lag_number)
fundamentals['intexp_lag'] = fundamentals['intexp'].shift(lag_number)
fundamentals['roic_lag'] = fundamentals['roic'].shift(lag_number)
fundamentals['evebitda_lag'] = fundamentals['evebitda'].shift(lag_number)
fundamentals['roe_lag'] = fundamentals['roe'].shift(lag_number)
fundamentals['netinc_lag'] = fundamentals['netinc'].shift(lag_number)
fundamentals['debtc_lag'] = fundamentals['debtc'].shift(lag_number)
fundamentals['debtnc_lag'] = fundamentals['debtnc'].shift(lag_number)
fundamentals['deferredrev_lag'] = fundamentals['deferredrev'].shift(lag_number)
fundamentals['depamor_lag'] = fundamentals['depamor'].shift(lag_number)
fundamentals['inventory_lag'] = fundamentals['inventory'].shift(lag_number)
fundamentals['assetturnover_lag'] = fundamentals['assetturnover'].shift(lag_number)
fundamentals['workingcapital_lag'] = fundamentals['workingcapital'].\
    shift(lag_number)
fundamentals['capex_to_sales_lag'] = fundamentals['capex_to_sales'].\
    shift(lag_number)
fundamentals['YoY_Rev_Growth_lag'] = fundamentals['YoY_Rev_Growth'].\
    shift(lag_number)
fundamentals['YoY_NetInc_Growth_lag'] = fundamentals['YoY_NetInc_Growth'].\
    shift(lag_number)
fundamentals['YoY_NCFO_Growth_lag'] = fundamentals['YoY_NCFO_Growth'].\
    shift(lag_number)
fundamentals['Balance_Sheet_Accruals_Ratio_lag'] = \
    fundamentals['Balance_Sheet_Accruals_Ratio'].shift(lag_number)
fundamentals['Sector_YoY_Earnings_Average_lag'] = \
    fundamentals['Sector_YoY_Earnings_Average'].shift(lag_number)
fundamentals['Ratio_Comp_YoY_Earn_Rate_vs_Sector_lag'] = \
    fundamentals['Ratio_Comp_YoY_Earn_Rate_vs_Sector'].shift(lag_number)
fundamentals['Sector_YoY_Revenue_Average_lag'] = \
    fundamentals['Sector_YoY_Revenue_Average'].shift(lag_number)
fundamentals['Ratio_Comp_YoY_Reve_Rate_vs_Sector_lag'] = \
    fundamentals['Ratio_Comp_YoY_Reve_Rate_vs_Sector'].shift(lag_number)

#filter out companies with less than $1 billion market cap based on previous quarter
fundamentals = fundamentals[fundamentals['marketcap'] >= 1*10e8]

features = fundamentals[['calendardate', 'fcf',
                         'netinc_lag',
                         'Revenue_lag',
                         'capex_lag',
                         'de_lag',
                         'ebitda_lag',
                         'opinc_lag',
                         'taxexp_lag',
                         'gp_lag',
                         'rnd_lag',
                         'sgna_lag',
                         'intexp_lag',
                         'roic_lag',
                         'roe_lag',
                         'evebitda_lag',
                         'debtc_lag',
                         'debtnc_lag',
                         'deferredrev_lag',
                         'depamor_lag',
                         'inventory_lag',
                         'assetturnover_lag',
                         'capex_to_sales',
                         'workingcapital_lag',
                         'YoY_Rev_Growth_lag',
                         'YoY_NetInc_Growth_lag',
                         'YoY_NCFO_Growth_lag',
                         'Balance_Sheet_Accruals_Ratio_lag',
                         'Sector_YoY_Earnings_Average_lag',
                         'Ratio_Comp_YoY_Earn_Rate_vs_Sector_lag',
                         'Sector_YoY_Revenue_Average_lag',
                         'Ratio_Comp_YoY_Reve_Rate_vs_Sector_lag'
                        ]]

#use the dates as the index which will be needed when merging
features.index = features.calendardate

#join macroeconomic variables to dataframe
features = features.join(Macro_Econ_Features_quarterly)

#change inf to NAs
features = features.replace([np.inf, -np.inf], np.nan)

#get rid of NAs
features = features.dropna()

#check no NAs or inf
features.isna().sum()
features.min()
features.max()

#######################Prep the Data for use in RF############################

#Use numpy to convert to arrays

# Labels are the values we want to predict
labels = np.array(features['fcf'])
# Remove the labels from the feature and the date
features_array = features.drop(['fcf', 'calendardate'], axis=1)
# Saving feature names for later use
feature_list = list(features_array.columns)
# Convert to numpy array
features_array = np.array(features_array)

# Using Skicit-learn to split data into training and testing sets, but in this case
#it is a validation set

# Split the data into training and testing sets
train_features, validation_features, train_labels, validation_labels = \
    train_test_split(features_array, labels, test_size=0.25, random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', validation_features.shape)
print('Testing Labels Shape:', validation_labels.shape)

number_features = train_features.shape[1]

# Instantiate model with 1000 decision trees
rf_PFCF = RandomForestRegressor(n_estimators=1000, random_state=42, \
                           criterion="mse", max_depth=None, \
                           max_features=round(number_features/3))

# Train the model on training data
rf_PFCF.fit(train_features, train_labels)

################Accuracy Scoring#########################################

# save the model to disk
path_to_file = (
        r'/Users/downey/coding/Python/Scripts/'
        'EPAT Project RF Models, Code, and Results/'
        'Random Forest Models/'
        )

Sector_RF_Model = 'RF_Model_PFCF_'+test_sector
filename = path_to_file + Sector_RF_Model + '.sav'
pickle.dump(rf_PFCF, open(filename, 'wb'))

# Use the forest's predict method on the test data
predictions = rf_PFCF.predict(train_features)

# Calculate the absolute errors
errors = abs(predictions - train_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')

MAE = sum(abs(errors))/errors.size
Average_Earnings = sum(abs(train_labels))/train_labels.size
train_MAE_Percentage = MAE/Average_Earnings

#validation set

# Use the forest's predict method on the test data
predictions_val = rf_PFCF.predict(validation_features)

#predicting a few earnings
rf_PFCF.predict(validation_features[-2:-1])

# Calculate the absolute errors
errors = abs(predictions_val - validation_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')

MAE = sum(abs(errors))/errors.size
Average_Earnings = sum(abs(validation_labels))/validation_labels.size
validation_MAE_Percentage = MAE/Average_Earnings

#compute R squared for data
score_R2_train = rf_PFCF.score(train_features, train_labels) #train set
score_R2_validation = rf_PFCF.score(validation_features, validation_labels) #validation set

R2_df = pd.DataFrame({'R^2 Score Train': score_R2_train,
                      'R^2 Score Validation': score_R2_validation}, index=[test_sector])
print(R2_df)

Sector_R2_score = 'IS R^2 Score PFCF'+test_sector

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models,'
    ' Code, and Results/R2 Score/'
    )
output_name = path_to_file + Sector_R2_score  + '.csv'
R2_df.to_csv(output_name)


#Write the Mean Absolute Error to csv file for each sector
MAE = pd.DataFrame([train_MAE_Percentage, validation_MAE_Percentage])
MAE = MAE.rename(index={0: "Train", 1: "Validation"})
MAE.columns = ['Mean Absolute Error (MAE)']

Sector_MAE = 'MAE PFCF '+test_sector

path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, '
    'Code, and Results/Mean Absolute Error/'
    )
output_name = path_to_file + Sector_MAE + '.csv'
MAE.to_csv(output_name)

#write feature list to txt file and save
with open('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, '
          'Code, and Results/feature_list_PFCF.txt', 'wb') as fp:
    pickle.dump(feature_list, fp)

##########################Forming the Forecast#############################3

fundamentals_2 = pd.DataFrame(fundamentals)

fundamentals_2.index = fundamentals_2.calendardate

features_2 = fundamentals_2[['calendardate', 'fcf', 'ticker', 'price',
                             'marketcap',
                             'netinc_lag',
                             'Revenue_lag',
                             'capex_lag',
                             'de_lag',
                             'ebitda_lag',
                             'opinc_lag',
                             'taxexp_lag',
                             'gp_lag',
                             'rnd_lag',
                             'sgna_lag',
                             'intexp_lag',
                             'roic_lag',
                             'roe_lag',
                             'evebitda_lag',
                             'debtc_lag',
                             'debtnc_lag',
                             'deferredrev_lag',
                             'depamor_lag',
                             'inventory_lag',
                             'assetturnover_lag',
                             'capex_to_sales',
                             'workingcapital_lag',
                             'YoY_Rev_Growth_lag',
                             'YoY_NetInc_Growth_lag',
                             'YoY_NCFO_Growth_lag',
                             'Balance_Sheet_Accruals_Ratio_lag',
                             'Sector_YoY_Earnings_Average_lag',
                             'Ratio_Comp_YoY_Earn_Rate_vs_Sector_lag',
                             'Sector_YoY_Revenue_Average_lag',
                             'Ratio_Comp_YoY_Reve_Rate_vs_Sector_lag'
                             ]]

#join macroeconomic variables to dataframe
features_2 = features_2.join(Macro_Econ_Features_quarterly)

features_3 = pd.DataFrame(features_2)

predictions_val = rf_PFCF.predict(validation_features)
predictions_train = rf_PFCF.predict(train_features)

forecasted_values = np.append(predictions_train, predictions_val)

#forecasted_12mo_earnings
forecasted_12mo_earnings = pd.Series(forecasted_values, name='12 Mo Forecast FCF')

features_3 = features_3.reset_index()

features_4 = pd.concat((features_3, forecasted_12mo_earnings), axis=1, join='inner')

#create forward P/E
features_4['Forward FCF/P'] = features_4['12 Mo Forecast FCF']/features_4['marketcap']

#confirming that the estimates are aligned correctly
rf_PFCF.predict(train_features[1:2, :])
features_4['12 Mo Forecast FCF'].head()

rf_PFCF.predict(validation_features[-2:-1, :])
features_4['12 Mo Forecast FCF'].tail()

Data_for_Portfolio_PFCF = features_4[['calendardate', 'ticker',
                                      '12 Mo Forecast FCF',
                                      'Forward FCF/P']]


path_to_file = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/Random Forest Models/ROIC Forecasts/'
    )
output_name = path_to_file + test_sector + '_PFCF_Predictions.csv'
Data_for_Portfolio_PFCF.to_csv(output_name)

t1 = time.time()
print('It took ' + str((t1-t0)/60) + ' minutes to run the code')

os.system("say 'Your Price to Cash Flow Code is Done'")
#%%
