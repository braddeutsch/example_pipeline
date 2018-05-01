import pandas as pd


def load_data(train_file, test_file):
    """Load test and train data into a DataFrame
    :return pd.DataFrame with ['test'/'train', features]"""

    # load train and test data
    data_train = pd.read_csv(train_file)
    data_test = pd.read_csv(test_file)

    # concat and label
    data_out = pd.concat([data_train, data_test], keys=['train', 'test'])

    return data_out


def preprocess_data(df, target_map):
    """
    General element-wise cleanup of full data set
    :return: df, cleaned data
    """

    # Strip whitespace
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].apply(lambda x: str.strip(x))

    # map target to binary
    df['target'] = df['target'].map(target_map)

    return df
