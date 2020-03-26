#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:15:45 2020

@author: downey
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import quandl
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pickle
import os

from IPython.display import set_matplotlib_formats
set_matplotlib_formats( 'pdf','png')
pd.set_option('display.max_columns',110)


'''This file shows feature importance, example trees, forecast vs. actual'''

##############Show Feature Importance#########################################
#%%
sectors = ['Healthcare', 'Basic Materials', 'Financial Services','Consumer Cyclical',
'Technology', 'Industrials','Energy','Real Estate', 'Consumer Defensive', 
'Communication Services', 'Utilities']

 #with open("Feature_list.txt", "wb") as fp:   #Pickling
    #pickle.dump(feature_list, fp)

os.chdir('/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code, and Results/')

with open("Feature_list.txt", "rb") as fp:   # Unpickling
    feature_list = pickle.load(fp)

feature_list

results = pd.DataFrame()
results = pd.concat([results, pd.DataFrame(feature_list)], axis=1,ignore_index = True)
Factor = 'PE_' #Rotate through 'EVEBITDA_' , 'ROIC_','PE_', and 'PFCF_'

for SECTOR in sectors:
  
    Sector_RF_Model = 'RF_Model_' + Factor + SECTOR
    Sector_RF_Model
    path_to_file=r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code, and Results/Random Forest Models/'
    filename = path_to_file + Sector_RF_Model + '.sav'
    
    loaded_model = pickle.load(open(filename, 'rb'))
     
    # Get numerical feature importances
    importances = list(loaded_model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    #feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    feature_df = pd.DataFrame(feature_importances)
    feature_df.columns = [['Features',SECTOR]]
    feature_df = feature_df.drop(columns = 'Features')
    results = pd.concat([results, feature_df], axis=1,ignore_index = True)

column_names = ['Features','Healthcare', 'Basic Materials', 'Financial Services','Consumer Cyclical',
'Technology', 'Industrials','Energy', 'Real Estate','Consumer Defensive', 
'Communication Services', 'Utilities']
results.columns = [column_names]

results.head()

path_to_file_2=r'/Users/downey/coding/Python/Scripts/EPAT Project RF Models, Code, and Results/Feature Importance/'
output_name = path_to_file_2 + Factor + '_feature_importance.csv'
results.to_csv(output_name)
#%%
feature_df

# Set the style
plt.style.use('fast')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

#####################Create Random Forest Tree Example########################

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = loaded_model.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, \
                rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

#####################Chart of Actual Vs. Predicted Earnings###################
#%%
#Graph predicated and actual earnings in validation set

#Run Out of Sample RF PE python file and use data here up until line 421

plt.plot(OOS_labels, 'b-')
plt.plot(predictions_val,'ro')
plt.xticks(rotation = '60'); 

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

red_patch = mpatches.Patch(color='red', label='Prediction')
blue_patch = mpatches.Patch(color='blue', label='True Value')
plt.legend(handles=[red_patch,blue_patch])
# Graph labels
plt.xlabel('Data'); plt.ylabel('Earnings (Billions USD)'); plt.title('Basic Materials Out of Sample');
plt.show()
#%%)

######################Smaller Tree png#######################################3

# Limit depth of tree to 3 levels for easier visualiztion
rf_small = RandomForestRegressor(n_estimators=100, max_depth = 8)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[2]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', \
                feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');
