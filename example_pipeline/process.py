#!/usr/bin/env python
"""
Example script for training and testing a model. Loads data, pre-processes, builds and executes a gridsearch pipeline,
produces and records test results.
"""

from datetime import datetime as dt
import pandas as pd
from sklearn.metrics import classification_report
from data.make_dataset import load_data, preprocess_data
from models.train_model import make_pipeline
from models.save_results import save_results


# CONFIG ####################################################################################

# file locations
train_file = 'c://data/example_pipeline/adult.data.txt'
test_file = 'c://data/example_pipeline/adult.test.txt'
var_codes_file = '../data/adult.vars.csv'

# name of the target column in your data
target_name = 'target'

# create save path
timestamp_str = dt.now().strftime('%Y%m%d_%H%M%S')
save_path = 'models/' + timestamp_str + '/'

# LOAD AND PRE-PROCESS DATA ########################################################################

# Load variable codes
var_codes = pd.read_csv(var_codes_file)
# Load data
data_all = load_data(train_file, test_file)

# pre-process data. Careful to only do element-wise operations.
target_map = {'>50K.': 1, '>50K': 1, '<=50K.': 0, '<=50K': 0}
data_clean = preprocess_data(data_all, target_map)

# split data back into test and train
df_train = data_clean.ix['train']
df_test = data_clean.ix['test']

# CREATE PIPELINE ###################################################################################

# make a list of features to include
features_to_include = var_codes[var_codes['include_gbf'] == 1]['var_name'].values
# build pipeline
pipeline = make_pipeline(features_to_include)

# TRAIN #############################################################################################

# split features from target
X = df_train[df_train.columns.drop('target')]
y = df_train['target']

# Fit the pipeline
model = pipeline.fit(X, y).best_estimator_

# TEST AND EVAL #####################################################################################

# separate features from target for test set
X_test = df_test.drop('target', axis=1)
y_test = df_test['target']
# make predictions
test_predictions = model.predict(X_test)
# build report
report = classification_report(y_test, test_predictions)

# SAVE AND REPORT ###################################################################################

save_results(save_path, model, report)
print("Results and model saved in %s" % save_path)
