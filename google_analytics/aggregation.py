import os
import warnings
import numpy as np
import pandas as pd


def load_train_test_dataframes(data_dir, train_file_name='preprocessed_train.csv',
                               test_file_name='preprocessed_test.csv', nrows_train=None, nrows_test=None):
    """ Load the train and test DataFrames resulting from preprocessing. """
    train = pd.read_csv(os.path.join(data_dir, train_file_name),
                        dtype={"fullVisitorId": str},
                        nrows=nrows_train)
    test = pd.read_csv(os.path.join(data_dir + test_file_name),
                       dtype={"fullVisitorId": str},
                       nrows=nrows_test)
    return train, test


def one_hot_encode_categoricals(data, categorical_columns):
    """ Transform categorical data to one-hot encoding and
        aggregate per customer.

    params
    ------
    data: DataFrame to transform.
    categorical_columns: array of column names indicating the
        columns to transform to one-hot encoding.

    notes
    -----
    The resulting columns are named as
    <original column name>_<original value>.

    Assumes the column fullVisitorId as grouper

    return
    ------
    The one-hot encoded DataFrame.
    """
    for col in categorical_columns:
        if data[col].dtypes in ["int64", "float64"]:
            warnings.warn("Column {} converted to from numeric to category".format(col))
            data[col] = data[col].astype("category")

    encoded = pd.get_dummies(data[categorical_columns],
                             prefix=categorical_columns)
    encoded["fullVisitorId"] = data["fullVisitorId"]
    return encoded.groupby("fullVisitorId").sum()


def summarize_numerical_data(data, cols_to_describe, cols_to_sum):
    """ Aggregate the numerical columns in the data per customer by
        summarizing / describing their values.

    params
    ------
    data: the DataFrame to aggregate.
    cols_to_describe: array-like of column names for which to compute
        descriptive measures such as mean, min, max, std, sum.
    cols_to_sum: array-like of column names for which to only compute
        the sum.

    notes
    -----
    Aggregates by the column "fullVisitorId", so this column must
    be present.

    return
    ------
    The aggregated data with one row per customer and several columns for
    every column in the original data: min, max, mean, std, and sum for the
    columns in 'cols_to_describe' and the sum for the columns in 'cols_to_sum'
    """
    # describe columns
    data_describe = data.groupby('fullVisitorId')[cols_to_describe] \
                        .agg(['min', 'max', 'mean', 'std', 'sum'])
    data_describe.columns = ['_'.join(col) for col in data_describe.columns]
    data_describe[data_describe.columns[data_describe.columns.str.contains('std')]] \
        .fillna(0, inplace=True)

    # sum columns if specified
    if cols_to_sum is not None:
        data_sum = data.groupby(['fullVisitorId'])[cols_to_sum].sum().add_suffix("_sum")
        return pd.concat([data_describe, data_sum], axis=1)
    else:
        return data_describe


def get_means_of_booleans(data, boolean_cols):
    """ Put boolean_cols of the data in a uniform format and
        compute the mean per customer.

    params
    ------
    data: The DataFrame.
    boolean_cols: array-like of column names with boolean values to
        process.

    return
    ------
    DataFrame with a row for each customer and columns presenting
    the percentage True for every column in boolean_cols.
    """
    # Some values are given in True/False, some in 1/NaN, etc.
    # Here we unify this to 1 and 0.
    data[boolean_cols] *= 1
    data[boolean_cols] = data[boolean_cols].fillna(0)
    # Calculate the percentage of 1s for each fullVisitorId
    data_bool = data.groupby(['fullVisitorId'])[boolean_cols].mean()
    data_bool = data_bool.add_suffix('_avg')
    return data_bool


def add_datetime_features(data, date_col="date"):
    """ Calculate the time between first and last visit.

    params
    ------
    data: The DataFrame.
    date_col: String, the name of the column in 'data' that
        represents the date of the visit.

    return
    ------
    DataFrame with a row for each fullVisitorId in data and
    columns with the dates of their first and last visits.
    """
    data[date_col] = pd.to_datetime(data[date_col])
    data_date = data.groupby(['fullVisitorId'])[date_col] \
                    .agg(['min', 'max'])

    data_date['days_first_to_last_visit'] = \
        (data_date['max'] - data_date['min']).dt.days

    del data_date['max']
    del data_date['min']

    return data_date


def add_mean_time_between_visits(df):
    """ Add the mean inter-visit time to the DataFrame.

    params
    ------
    df: DataFrame where every row represents a customer.

    return
    ------
    The same DataFrame with an additional column 'mean_intervisit_time'
    that holds the mean time between two consecutive visits of the
    customer.
    """
    safe = df["visitNumber_nunique"] > 1
    intervisit_time = np.zeros(len(df))
    intervisit_time[safe] = df["days_first_to_last_visit"][safe] \
        / (df["visitNumber_nunique"][safe]-1)
    df["mean_intervisit_time"] = intervisit_time
    return df


def aggregate_data_per_customer(data):
    """ Aggregate the data per customer by one-hot encoding categorical
        variables and summarizing numerical variables.

    params
    ------
    data: DataFrame
        The data to aggregate.
    target_col_present: boolean
        Indicates whether the target column 'target' is in the data,
        so put this to True for the train data and False for test.

    return
    ------
    The aggregated DataFrame with one row per customer and
    a shit load of columns.
    """
    def get_mode(x):
        try:
            return x.mode()[0]
        except KeyError:
            return np.nan
        except IndexError:
            return np.nan

    # specify what to do with each column
    OHE = ['channelGrouping', 'browser', 'operatingSystem',
           'continent', 'country', 'subContinent', 'adContent',
           'adwordsClickInfo.adNetworkType', 'adwordsClickInfo.page',
           'adwordsClickInfo.slot', 'campaign', 'medium', 'WoY', 'month',
           'quarterMonth', 'weekday', 'visitHour', 'source_cat']
    cat_nunique = ['networkDomain', 'referralPath', 'keyword', 'source']
    unique_value = ['city', 'metro', 'deviceCategory', 'region']
    num_nunique = ['visitNumber']
    num_describe = ['hits', 'pageviews', 'keyword.mistakes_Google', 'keyword.mistakes_YouTube']
    booleans = ['isMobile', 'bounces', 'newVisits', 'adwordsClickInfo.isVideoAd',
                'isTrueDirect', 'keyword.isGoogle', 'keyword.isYouTube']

    if 'target' in data.columns:
        num_sum = ['target']
    else:
        num_sum = None

    # pre-process categorical data: one-hot encode
    print("Summarizing the categorical variables...")
    data_categoricals = one_hot_encode_categoricals(data, OHE+unique_value)
    # for the columns in 'unique_values', there is only one value per customer
    # and the rest is NaN, so we need just a 1 for the value and remove the NaN columns
    data_categoricals = \
        data_categoricals.loc[:, ~data_categoricals.columns.str.contains("NaN")]
    for col in unique_value:
        data_categoricals.loc[:, data_categoricals.columns.str.contains(col)] = \
            np.minimum(1, data_categoricals.loc[:, data_categoricals.columns.str.contains(col)])

    # categorical data with large numbers of unique values are dealt
    # with by only taking the number of unique values per customer
    data_diff = data.groupby(['fullVisitorId'])[cat_nunique] \
                    .nunique().add_suffix('_diff')

    # pre-process numerical data
    print("Summarizing the numerical variables...")
    # first calculate number of visits in the dataset (unique visitNumber)
    # and the total number of visits of the customer (max visitNumber)
    data_nunique = data.groupby(['fullVisitorId'])[num_nunique].agg(['nunique', 'max'])
    data_nunique.columns = ['_'.join(col) for col in data_nunique.columns]

    # describe some columns in more detail by getting the
    # the min, max, mean, and std;
    # plus the sum of the target variable (which is what we want to predict)
    data_numericals = summarize_numerical_data(data, num_describe, num_sum)

    # handle booleans by taking the mean
    data_bools = get_means_of_booleans(data, booleans)

    # create datetime features: time between first and last visit
    # and mean time between consecutive visits
    data_dates = add_datetime_features(data)

    # merge
    print("Putting it all together..")
    df = pd.concat([data_categoricals, data_diff, data_nunique,
                    data_numericals, data_bools, data_dates], axis=1)

    # add mean time between visits
    df = add_mean_time_between_visits(df)
    print("Done")

    return df


def ohe_explicit(df):
    """Partly one hot encodes specific categorical features.

    Instead of one hot encoding categorical columns with many distinct values (200+) we instead only create
    boolean columns corresponding to specific values based on two criteria:
        * Statistical Significance. These values should not be rare, else we are overfitting.
        * Predictive Power. These values should have conditional averages considerably different from the global
          target mean.

    The choice of features and values has been made manually during EDA and is subject to change in case we come
    up with better insights in the future.

    Examples
    --------
    >>> train = load("../data/train.csv")
    >>> check = ohe_explicit(train)
    >>> check.head()

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe including the raw categorical features for Country and City.

    Returns
    -------
    pd.DataFrame
        The original categorical columns are replaced by one hot columns for specific values only.

    """
    countries = ["United States"]
    for country in countries:
        df["country_" + country] = df["country"].apply(lambda c: c == country)

    df.drop("country", axis=1, inplace=True)

    cities = ["New York", "Chicago", "Austin", "Seattle", "Palo Alto", "Toronto"]
    for city in cities:
        df["city_" + city] = df["city"].apply(lambda c: c == city)

    df.drop("city", axis=1, inplace=True)
    return df
