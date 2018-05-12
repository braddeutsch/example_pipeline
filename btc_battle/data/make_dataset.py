import pandas as pd
from dateutil import parser
import numpy as np


def load_data(_file, pct_split):
    """Load test and train data into a DataFrame
    :return pd.DataFrame with ['test'/'train', features]"""

    # load train and test data
    data = pd.read_csv(_file)

    # split into train and test using pct_split
    # data_train = ...
    # data_test = ...

    # concat and label
    # data_out = pd.concat([data_train, data_test], keys=['train', 'test'])

    data_out = data

    return data_out


def preprocess_data(df):
    """
    General element-wise cleanup of full data set
    :return: df, cleaned data
    """

    # convert time to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    return df


from datetime import timedelta
import numpy as np
import pandas as pd


def create_target(df):

    def btc_target(row):
        time_window = df['price'][(df['time'] > row) &
                                  (df['time'] <= row + timedelta(minutes=30))].values

        # return null if nothing is within this time window
        if len(time_window) == 0:
            return None

        # calculate thresholds
        minY = np.min(time_window)
        maxY = np.max(time_window)
        currentY = time_window[0]
        thresh_low = currentY * (1 - 0.0025)
        thresh_hi = currentY * (1 + 0.0025)

        # determine which thresholds are exceeded
        exceeds_low = minY <= thresh_low
        exceeds_hi = maxY >= thresh_hi

        # do Boolean logic to create a target. {neither: 0, high: 1, low: 0, both: 2}
        if (not exceeds_low) and (not exceeds_hi):
            target = 0
        elif exceeds_low and (not exceeds_hi):
            target = 1
        elif (not exceeds_low) and exceeds_hi:
            target = 2
        else:
            target = 3

        return target

    df['target'] = df['time'].apply(btc_target)

    return df


def forward_window_vals(time_col, val_col):
    return