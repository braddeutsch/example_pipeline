
from datetime import datetime as dt
import pandas as pd
from sklearn.metrics import classification_report
from data.make_dataset import preprocess_data
from models.train_model import make_pipeline
from models.save_results import save_results


# parameters
train_file = 'c://data/example_pipeline/adult.data.txt'
test_file = 'c://data/example_pipeline/adult.test.txt'
var_codes_file = 'c://data/example_pipeline/adult.vars.csv'
target_name = 'target'

timestamp_str = dt.now().strftime('%Y%m%d_%H%M%S')
save_path = 'models/' + timestamp_str + '/'

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

# CREATE PIPELINE ###################################################################################

features_to_include = df_train.columns.drop('target')
pipeline = make_pipeline(features_to_include)

# TRAIN #############################################################################################

model = pipeline.fit(df_train[df_train.columns.drop('target')], df_train['target']).best_estimator_

# TEST AND EVAL #####################################################################################

test_predictions = model.predict(df_test.drop('target', axis=1))
report = classification_report(df_test['target'], test_predictions)

# SAVE AND REPORT ###################################################################################

save_results(save_path, model, report)
