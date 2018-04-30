from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn_pandas import DataFrameMapper


def featurize_cols(features):

    transformations = [
        (['age'], StandardScaler()),
        (['workclass'], LabelBinarizer()),
        (['fnlwgt'], StandardScaler()),
        (['education'], LabelBinarizer()),
        (['education-num'], LabelBinarizer()),
        (['marital-status'], LabelBinarizer()),
        (['occupation'], LabelBinarizer()),
        (['relationship'], LabelBinarizer()),
        (['race'], LabelBinarizer()),
        (['sex'], LabelBinarizer()),
        (['capital-gain'], StandardScaler()),
        (['capital-loss'], StandardScaler()),
        (['hours-per-week'], StandardScaler()),
        (['native-country'], LabelBinarizer())
    ]

    return DataFrameMapper([t for t in transformations if t[0][0] in features], df_out=False)
