import json
import os
import time

import pandas as pd
import numpy as np


def load(path, nrows=None):
    """ Load Google analytics data from JSON into a Pandas.DataFrame. """
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},
                     nrows=nrows)

    # Normalize JSON columns
    for column in JSON_COLUMNS:
        column_as_df = pd.io.json.json_normalize(df[column])
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    # Parse date
    df['date'] = df['date'].apply(lambda x: pd.datetime.strptime(str(x), '%Y%m%d'))
    print("Loaded file {}\nShape is: {}".format(path, df.shape))
    return df


def process(train, test):
    """ Perform basic preprocessing of Google analytics data. """

    print("Dropping constant columns...")
    # Remove columns with constant values.
    const_cols = [c for c in train.columns if train[c].nunique(dropna=False) == 1]
    train = train.drop(const_cols, axis=1)
    test = test.drop(const_cols, axis=1)

    # Cast target
    train["transactionRevenue"] = train["transactionRevenue"].fillna(0).astype(float)
    train["target"] = np.log(train["transactionRevenue"] + 1)
    del train["transactionRevenue"]

    train_len = train.shape[0]
    merged = pd.concat([train, test], sort=False)

    # Change values as "not available in demo dataset", "(not set)",
    # "unknown.unknown", "(not provided)" to nan.
    list_missing = ["not available in demo dataset", "(not provided)",
                    "(not set)", "<NA>", "unknown.unknown",  "(none)"]
    merged = merged.replace(list_missing, np.nan)

    # Create some features.
    merged['diff_visitId_time'] = merged['visitId'] - merged['visitStartTime']
    merged['diff_visitId_time'] = (merged['diff_visitId_time'] != 0).astype(int)
    del merged['visitId']
    del merged['sessionId']

    print("Generating date columns...")
    merged['WoY'] = merged['date'].apply(lambda x: x.isocalendar()[1])
    merged['month'] = merged['date'].apply(lambda x: x.month)
    merged['quarterMonth'] = merged['date'].apply(lambda x: x.day // 8)
    merged['weekday'] = merged['date'].apply(lambda x: x.weekday())
    merged['visitHour'] = pd.to_datetime(merged['visitStartTime']
                                         .apply(lambda t: time.strftime('%Y-%m-%d %H:%M:%S',
                                                time.localtime(t)))) \
        .apply(lambda t: t.hour)

    del merged['visitStartTime']

    print("Finding total visits...")
    # This could be considered an information leak as I am including
    # information about the future when predicting the revenue of a
    # transaction. In reality, when looking at the 3rd visit we would
    # have no way of knowning that the user will actually shop X more
    # times (or if he will visit again at all). However since this
    # info also exists in the test set we might use it.
    total_visits = merged[["fullVisitorId", "visitNumber"]].groupby("fullVisitorId", as_index=False).max()
    total_visits.rename(columns={"visitNumber": "totalVisits"}, inplace=True)
    merged = merged.merge(total_visits)

    print("Splitting back...")
    train = merged[:train_len]
    test = merged[train_len:]
    return train, test


def preprocess_and_save(data_dir, nrows_train=None, nrows_test=None):
    """ Preprocess and save the train and test data as DataFrames. """
    train = load(os.path.join(data_dir, "train.csv"), nrows=nrows_train)
    test = load(os.path.join(data_dir, "test.csv"), nrows=nrows_test)

    train, test = process(train, test)
    train.to_csv(os.path.join(data_dir, "preprocessed_train.csv"), index=False, encoding="utf-8")
    test.to_csv(os.path.join(data_dir, "preprocessed_test.csv"), index=False, encoding="utf-8")


def keep_intersection_of_columns(train, test):
    """ Remove the columns from test and train set that are not in
        both datasets.

    params
    ------
    train: pd.DataFrame containing the train set.
    test: pd.DataFrame containing the test set.

    return
    ------
    train and test where train.columns==test.columns by
    keeping only columns that were present in both datasets.
    """
    shared_cols = list(set(train.columns).intersection(set(test.columns)))
    return train[shared_cols], test[shared_cols]
