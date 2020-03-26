#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:15:45 2020

@author: downey
"""

import pandas as pd

pd.set_option('display.max_columns', 110)

####    This file combines the r2 score of sectors    ####

##############Show Feature Importance#########################################

SECTORS = ['Healthcare',
           'Basic Materials',
           'Financial Services',
           'Consumer Cyclical',
           'Technology',
           'Industrials',
           'Real Estate',
           'Energy',
           'Consumer Defensive',
           'Communication Services',
           'Utilities']

FACTOR = 'PE' #Rotate through the four factors
PATH_TO_FILE = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/R2 Score/'
    )
RESULTS = pd.DataFrame()
for sector in SECTORS:

    factor = 'IS R^2 Score ' + FACTOR
    specific_path = PATH_TO_FILE + factor + sector + '.csv'
    IS_R2 = pd.read_csv(specific_path)


    factor2 = 'OOS R^2 Score ' + FACTOR
    specific_path2 = PATH_TO_FILE + factor2 + sector + '.csv'
    OS_R2 = pd.read_csv(specific_path2)

    df = IS_R2.merge(OS_R2, how='inner', on=['Unnamed: 0'])

    dfT = df.T
    RESULTS = RESULTS.append(dfT, ignore_index=False)

PATH_TO_FILE_2 = (
    r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code,'
    ' and Results/R2 Score/'
    )

OUTPUT_NAME = PATH_TO_FILE_2 + FACTOR + '_R2_Combined_scores.csv'
RESULTS.to_csv(OUTPUT_NAME)
