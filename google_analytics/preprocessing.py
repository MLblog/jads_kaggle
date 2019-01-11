import json
import os
import time

import pandas as pd
import numpy as np

import Levenshtein
import re


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


def create_train_test(train, test, start_x_train='2016-08-01', end_x_train='2017-10-16',
                      start_y_train='2017-12-01', end_y_train='2018-02-01', start_x_test='2017-08-01',
                      end_x_test='2018-10-16', drop_users=True):
    """ Splits the original data into x_train, y_train, and x_test based on the date of the visits.
    Besides that, it drops visitors that visit once in the training periods and never bought.
    """
    def filter_users(data):
        count_visits = data.groupby("fullVisitorId").size().reset_index(name='count')
        single_visit_ids = count_visits[count_visits["count"] == 1]["fullVisitorId"]
        sum_spent = data[["fullVisitorId", "transactionRevenue"]].groupby("fullVisitorId", as_index=False).sum()
        no_buy_ids = sum_spent[sum_spent["transactionRevenue"] == 0]["fullVisitorId"]
        drop_ids = set(single_visit_ids).intersection(set(no_buy_ids))
        data = data[~data["fullVisitorId"].isin(drop_ids)]
        return data.copy(), drop_ids

    merged = pd.concat([train, test], sort=False)

    # Split the total dataset into x_train, y_train, and x_test
    x_train = merged[(merged["date"] >= start_x_train) & (merged["date"] < end_x_train)]
    y_train = merged[(merged["date"] >= start_y_train) & (merged["date"] < end_y_train)]
    x_test = merged[(merged["date"] >= start_x_test) & (merged["date"] < end_x_test)]

    # Drop visitors that visited once and never bought during the training period from the datasets
    x_train, drop_ids_train = filter_users(x_train)
    y_train = y_train[~y_train["fullVisitorId"].isin(drop_ids_train)].copy()
    x_test, _ = filter_users(x_test)

    return x_train, y_train, x_test


def process_x(train, test):
    """ Perform basic preprocessing of Google analytics data. """

    print("Dropping constant columns...")
    # Remove columns with constant values.
    const_cols = [c for c in train.columns if train[c].nunique(dropna=False) == 1]
    train = train.drop(const_cols, axis=1)
    test = test.drop(const_cols, axis=1)

    print("Finding total visits...")
    # This could be considered an information leak as I am including
    # information about the future when predicting the revenue of a
    # transaction. In reality, when looking at the 3rd visit we would
    # have no way of knowning that the user will actually shop X more
    # times (or if he will visit again at all). However since this
    # info also exists in the test set we might use it.
    # Update: in the new competition I think this is no data leakage anymore.
    total_visits = train[["fullVisitorId", "visitNumber"]].groupby('fullVisitorId').size().reset_index(name='totalVisits')
    train = train.merge(total_visits, on='fullVisitorId')
    total_visits = test[["fullVisitorId", "visitNumber"]].groupby('fullVisitorId').size().reset_index(name='totalVisits')
    test = test.merge(total_visits, on='fullVisitorId')

    train_len = train.shape[0]
    merged = pd.concat([train, test], sort=False)

    # Ensure correct train-test split
    merged["manual_index"] = np.arange(0, len(merged))

    # Cast target
    merged["transactionRevenue"] = merged["transactionRevenue"].fillna(0).astype(float)
    merged["target"] = np.log(merged["transactionRevenue"] + 1)
    del merged["transactionRevenue"]

    # Change values as "not available in demo dataset", "(not set)",
    # "unknown.unknown", "(not provided)" to nan.
    list_missing = ["not available in demo dataset", "(not provided)",
                    "(not set)", "<NA>", "unknown.unknown",  "(none)"]
    merged = merged.replace(list_missing, np.nan)

    # Create some features.
    print("Create some features...")
    merged['diff_visitId_time'] = merged['visitId'] - merged['visitStartTime']
    merged['diff_visitId_time'] = (merged['diff_visitId_time'] != 0).astype(int)
    del merged['visitId']
    del merged['sessionId']
    # Check whether there is refered to google or youtube in the search term and report the number of spelling mistakes made while searching
    merged['keyword.isGoogle'] = merged['keyword'].apply(lambda x: isSimilar(x, 'google', 2)[0])
    merged['keyword.mistakes_Google'] = merged['keyword'].apply(lambda x: isSimilar(x, 'google', 2)[1])
    merged['keyword.isYouTube'] = merged['keyword'].apply(lambda x: isSimilar(x, 'youtube', 3)[0])
    merged['keyword.mistakes_YouTube'] = merged['keyword'].apply(lambda x: isSimilar(x, 'youtube', 3)[1])
    # generalize referralPath
    merged['referralPath'] = merged['referralPath'].apply(lambda x: replace(x, '/yt/about/', '/yt/about/'))
    # generalize sources
    merged['source_cat'] = merged['source'].apply(lambda x: give_cat(x))

    print("Generating date columns...")
    merged['WoY'] = merged['date'].apply(lambda x: x.isocalendar()[1])
    merged['month'] = merged['date'].apply(lambda x: x.month)
    merged['quarterMonth'] = merged['date'].apply(lambda x: x.day // 8)
    merged['weekday'] = merged['date'].apply(lambda x: x.weekday())
    merged['year'] = merged['date'].apply(lambda x: x.year)
    merged['visitHour'] = pd.to_datetime(merged['visitStartTime']
                                         .apply(lambda t: time.strftime('%Y-%m-%d %H:%M:%S',
                                                time.localtime(t)))) \
        .apply(lambda t: t.hour)

    del merged['visitStartTime']

    print("Splitting back...")
    merged.sort_values(by="manual_index", ascending=True, inplace=True)
    train = merged[:train_len]
    test = merged[train_len:]
    return train, test


def process_y(y_train):
    """ Perform basic preprocessing of Google analytics target data. """
    y_train = y_train[['fullVisitorId', 'transactionRevenue']].copy()
    # Cast target
    y_train["transactionRevenue"] = y_train["transactionRevenue"].fillna(0).astype(float)
    y_train["target"] = np.log(y_train["transactionRevenue"] + 1)
    del y_train["transactionRevenue"]

    return y_train


def preprocess_and_save(data_dir, nrows_train=None, nrows_test=None, start_x_train='2016-08-01',
                        end_x_train='2017-10-16', start_y_train='2017-12-01', end_y_train='2018-02-01',
                        start_x_test='2017-08-01', end_x_test='2018-10-16', drop_users=True):
    """ Preprocess and save the train and test data as DataFrames. """
    train = load(os.path.join(data_dir, "train.csv"), nrows=nrows_train)
    test = load(os.path.join(data_dir, "test.csv"), nrows=nrows_test)
    x_train, y_train, x_test = create_train_test(train, test, start_x_train, end_x_train,
                                                 start_y_train, end_y_train, start_x_test,
                                                 end_x_test, drop_users=drop_users)
    x_train, x_test = process_x(x_train, x_test)
    y_train = process_y(y_train)
    x_train.to_csv(os.path.join(data_dir, "preprocessed_x_train.csv"), index=False, encoding="utf-8")
    y_train.to_csv(os.path.join(data_dir, "preprocessed_y_train.csv"), index=False, encoding="utf-8")
    x_test.to_csv(os.path.join(data_dir, "preprocessed_x_test.csv"), index=False, encoding="utf-8")


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


def replace(string, regex, replacement):
    """ replace an original string when a regular expression is in this string

    params
    ------
    string: The original string
    regex: Regular expression
    replacement: The replacing string

    return
    ------
    When the regular expression is in the original string, the replacement is returned.
    Else the original string is returns
    """
    regex = re.compile(regex)
    if pd.isnull(string):
        return string
    else:
        if regex.search(string):
            return replacement
        else:
            return string


def isSimilar(string, aimed_string, threshold):
    """ Indicates whether string is similar to aimed_string including the levenshtein distance

    params
    ------
    string: The original string
    aimed_string: The aimed string
    threshold: The maximum levenshtein distance

    return
    ------
    label: boolean
        Is True when the levenshtein distance between string and aimed_string
        is smaller or equal to the threshold. Else it returns False.
    score: int
        The levenshtein distance between string and aimed_string
    """
    if pd.isnull(string):
        return np.nan, np.nan
    else:
        score = Levenshtein.distance(string.lower(), aimed_string)
        return_score = np.nan
        label = False
        if score > threshold:
            for word in string.split():
                score = Levenshtein.distance(word.lower(), aimed_string)
                if score <= threshold:
                    label = True
                    return_score = score
                    break
        else:
            label = True
            return_score = score
        return label, return_score


def give_cat(string):
    """ assigns the column 'source' to categories

    params
    ------
    string: The original source
    return
    ------
    The category to which the source belongs
    """
    regex_google = re.compile('google')
    regex_youtube = re.compile('youtube')
    regex_direct = re.compile('(direct)')
    regex_partners = re.compile('Partners')
    regex_search = re.compile('baidu|yahoo|bing|ask|duckduckgo|sogou|yandex|search')
    if pd.isnull(string):
        return string
    else:
        if regex_google.search(string):
            return 'google'
        elif regex_youtube.search(string):
            return 'youtube'
        elif regex_direct.search(string):
            return 'direct'
        elif regex_partners.search(string):
            return 'partners'
        elif regex_search.search(string):
            return 'other_search_engine'
        else:
            return np.nan
