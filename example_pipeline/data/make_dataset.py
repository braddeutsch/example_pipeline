import pandas as pd
from

def load_data(f):
    """Load data from a file into a Pandas dataframe"""

    d = pd.read_csv(f)

    return d


def train_test_split(df):

    