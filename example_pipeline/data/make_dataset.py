import pandas as pd


def load_data(f):
    """Load data from a file into a Pandas dataframe"""

    d = pd.read_csv(f, index_col=None)

    return d


def preprocess_data(df, target_map):

    # preprocessing (before split)
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].apply(lambda x: str.strip(x))

    # map target to binary
    df['target'] = df['target'].map(target_map)

    return df
