#!/usr/bin/env python
"""
Example script for training and testing a model. Loads data, pre-processes, builds and executes a gridsearch pipeline,
produces and records test results.
"""

from datetime import datetime as dt
import warnings
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.exceptions import DataConversionWarning
from data.make_dataset import load_data, preprocess_data
from modeling.train_model import make_pipeline
from modeling.save_results import save_results

__license__ = "MIT"

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# CONFIG ####################################################################################

# file locations
raw_data_path = '../data/2017-12-16-GDAX.finalfeatures.csv'
save_path = '../data/2017-12-16-GDAX.preprocessed.csv'

# LOAD AND PRE-PROCESS DATA #################################################################

print('Loading and pre-processing data...')
raw_data = pd.read_csv(raw_data_path)
preprocessed_data = preprocess_data(raw_data)

# Save
preprocessed_data.to_csv(save_path)
print('Done.')

# CREATE PIPELINE ###########################################################################

# make a list of features to include
features_to_include = var_codes[var_codes['include_gbf'] == 1]['var_name'].values
# build pipeline
pipeline = make_pipeline(features_to_include)

# TRAIN #####################################################################################

print('Running grid search...')

# split features from target
X = df_train[df_train.columns.drop('target')]
y = df_train['target']

# Fit the pipeline
model = pipeline.fit(X, y).best_estimator_

print('Done.')

# TEST AND EVAL #############################################################################

# separate features from target for test set
X_test = df_test.drop('target', axis=1)
y_test = df_test['target']
# make predictions
test_predictions = model.predict(X_test)
# build report
report = classification_report(y_test, test_predictions)

# SAVE AND REPORT ###########################################################################

save_results(save_path, model, report)
print("Results and model saved in %s" % save_path)
