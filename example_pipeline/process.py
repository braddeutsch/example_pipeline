
from datetime import datetime as dt

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report

from data.make_dataset import preprocess_data
from features.build_features import featurize_cols


# parameters
train_file = 'c://data/example_pipeline/adult.data.txt'
test_file = 'c://data/example_pipeline/adult.test.txt'
var_codes_file = 'c://data/example_pipeline/adult.vars.csv'
target_name = 'target'

timestamp_str = dt.now().strftime('%Y%m%d_%H%M%S')
save_path = 'results/' + timestamp_str + '/'

# load data
var_codes = pd.read_csv(var_codes_file)
data_train = pd.read_csv(test_file)
data_test = pd.read_csv(test_file)

# PREPROCESS DATA ##################################################################################

# concatenate test and train
data_all = pd.concat([data_train, data_test], keys=['train', 'test'])

# preprocess data. Careful to only do element-wise operations.
target_map = {'>50K.': 1, '<=50K.': 0}
data_clean = preprocess_data(data_all, target_map)

# split data back into test and train
df_train = data_clean.ix['train']
df_test = data_clean.ix['test']


# CREATE PIPELINES ###################################################################################

features_to_include = df_train.columns.drop('target')
pipeline = Pipeline([('featurize', featurize_cols(features_to_include)), ('gbf', GradientBoostingClassifier())])

# define cross-validation scheme
cv_folds = StratifiedKFold(n_splits=3)

# do a grid search to find best model
param_grid = dict(gbf__n_estimators=[5, 20, 50], gbf__subsample=[.6, .8], gbf__min_samples_leaf=[1, 3])

# define grid search object
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv_folds, scoring='accuracy')

# TRAIN #############################################################################################

model = grid_search.fit(df_train[df_train.columns.drop('target')], df_train['target']).best_estimator_

# TEST AND EVAL #####################################################################################

test_predictions = model.predict(df_test.drop('target', axis=1))
report = classification_report(df_test['target'], test_predictions)

# SAVE AND REPORT ###################################################################################


