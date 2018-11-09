import os
import warnings
import numpy as np
import pandas as pd


def load_train_test_dataframes(data_dir, x_train_file_name='preprocessed_x_train.csv',
                               y_train_file_name='preprocessed_y_train.csv',
                               x_test_file_name='preprocessed_x_test.csv', nrows_train=None,
                               nrows_test=None):
    """ Load the train and test DataFrames resulting from preprocessing. """
    x_train = pd.read_csv(os.path.join(data_dir, x_train_file_name),
                          dtype={"fullVisitorId": str},
                          nrows=nrows_train)

    y_train = pd.read_csv(os.path.join(data_dir, y_train_file_name),
                          dtype={"fullVisitorId": str})

    x_test = pd.read_csv(os.path.join(data_dir + x_test_file_name),
                         dtype={"fullVisitorId": str},
                         nrows=nrows_test)

    x_train['date'] = x_train['date'].apply(lambda x: pd.datetime.strptime(str(x), '%Y-%m-%d'))
    x_test['date'] = x_test['date'].apply(lambda x: pd.datetime.strptime(str(x), '%Y-%m-%d'))
    return x_train, y_train, x_test


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
    if categorical_columns == ['nr_months_ago']:
        encoded = pd.get_dummies(data[categorical_columns], prefix=['nr_visits'])
    else:
        encoded = pd.get_dummies(data[categorical_columns], prefix=categorical_columns)
    encoded["fullVisitorId"] = data["fullVisitorId"]
    return encoded.groupby("fullVisitorId").sum()


def summarize_numerical_data(data, cols, aggregation):
    """ Aggregate the numerical columns in the data per customer by
        summarizing / describing their values.

    Parameters
    ----------
    data: the DataFrame to aggregate.
    cols_to_describe: array-like of column names for which to compute
        descriptive measures such as mean, min, max, std, sum.
    cols_to_sum: array-like of column names for which to only compute
        the sum.

    Notes
    -----
    Aggregates by the column "fullVisitorId", so this column must
    be present.

    Returns
    -------
    The aggregated data with one row per customer and several columns for
    every column in the original data: min, max, mean, std, and sum for the
    columns in 'cols_to_describe' and the sum for the columns in 'cols_to_sum'
    """
    # describe columns
    data_describe = data.groupby('fullVisitorId')[cols] \
        .agg(aggregation)
    data_describe.columns = ['_'.join(col) for col in data_describe.columns]
    data_describe[data_describe.columns[data_describe.columns.str.contains('std')]] \
        .fillna(0, inplace=True)

    return data_describe


def get_means_of_booleans(data, boolean_cols):
    """ Put boolean_cols of the data in a uniform format and
        compute the mean per customer.

    Parameters
    ----------
    data: The DataFrame.
    boolean_cols: array-like of column names with boolean values to
        process.

    Returns
    -------
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


def nrmonths(start, end):
    """Fucntion that returns the number of months between two datetimes.

    Parameters
    ----------
    start, end: pd.Timestamp

    Returns
    -------
    Number of months difference between start and end rounded to integers.
    """
    return int(np.floor((end - start) / np.timedelta64(1, 'M')))


def get_dynamic(data, cols, method, timewindow='monthly'):
    """ Get values for columns over time.

    For now only the option to add dynamic features per month is described.

    Parameters
    ----------
    cols: list(str),
        The columns to get the dynamic values for.
    method: function,
        How to aggregate the values (e.g., 'sum', 'mean', 'count').
    timewindow: str, one of {'monthly'},
        Only monthly aggregation supported right now.

    Returns
    -------
    The aggregated data.
    """
    if timewindow == 'monthly':
        data_dynamic = pd.pivot_table(data, index='fullVisitorId', values=cols, columns='nr_months_ago',
                                      aggfunc=method, fill_value=0)
        data_dynamic.columns = [str(x[0]) + '_' + str(x[1]) + '_' + method for x in data_dynamic.columns]
    else:
        print('This is not a possible timewindow')
    return data_dynamic


def add_datetime_features(data, date_col="date"):
    """ Calculate the time between first and last visit.

    Parameters
    ----------
    data: The DataFrame.
    date_col: String, the name of the column in 'data' that
        represents the date of the visit.

    Returns
    -------
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

    Parameters
    ----------
    df: DataFrame where every row represents a customer.

    Returns
    -------
    The same DataFrame with an additional column 'mean_intervisit_time'
    that holds the mean time between two consecutive visits of the
    customer.
    """
    safe = df["totalVisits_mean"] > 1
    intervisit_time = np.zeros(len(df))
    intervisit_time[safe] = df["days_first_to_last_visit"][safe] / (df["totalVisits_mean"][safe] - 1)
    df["mean_intervisit_time"] = intervisit_time
    return df


def aggregate_data_per_customer(data, startdate_y, startdate_x):
    """ Aggregate the data per customer by one-hot encoding categorical
        variables and summarizing numerical variables.

    Parameters
    ----------
    data: DataFrame
        The data to aggregate.
    target_col_present: boolean
        Indicates whether the target column 'target' is in the data,
        so put this to True for the train data and False for test.

    Returns
    -------
    The aggregated DataFrame with one row per customer and
    a shit load of columns.
    """

    # Specify what to do with each column
    # Static
    OHE = ['channelGrouping', 'browser', 'deviceCategory', 'operatingSystem', 'city', 'continent',
           'country', 'metro', 'region', 'subContinent', 'adContent',
           'adwordsClickInfo.adNetworkType', 'adwordsClickInfo.page', 'adwordsClickInfo.slot',
           'campaign', 'medium', 'source_cat', 'weekday', 'visitHour']
    booleans = ['isMobile', 'adwordsClickInfo.isVideoAd', 'isTrueDirect', 'keyword.isGoogle', 'keyword.isYouTube']
    cat_nunique = ['networkDomain']
    num_mean = ['totalVisits', 'keyword.mistakes_Google', 'keyword.mistakes_YouTube']

    # Dynamic
    monthly_count = ['bounces', 'newVisits']
    monthly_mean = ['hits', 'pageviews', 'target']
    monthly_sum = ['hits', 'pageviews', 'target']

    # Pre-process static data
    print("Summarizing the static variables...")
    data_categoricals = one_hot_encode_categoricals(data, OHE)
    # For the columns in 'unique_values', there is only one value per customer
    # and the rest is NaN, so we need just a 1 for the value and remove the NaN columns
    data_categoricals = \
        data_categoricals.loc[:, ~data_categoricals.columns.str.contains("NaN")]

    # categorical data with large numbers of unique values are dealt
    # with by only taking the number of unique values per customer
    data_diff = data.groupby(['fullVisitorId'])[cat_nunique] \
        .nunique().add_suffix('_#diff')

    # handle booleans by taking the mean
    data_bools = get_means_of_booleans(data, booleans)

    # Describ num columns by getting the mean
    data_numericals = summarize_numerical_data(data, num_mean, ['mean'])

    # pre-process dynamic data
    print("Summarizing the dynamic variables...")
    startdate_y = pd.datetime.strptime(startdate_y, '%Y-%m-%d')
    startdate_x = pd.datetime.strptime(startdate_x, '%Y-%m-%d')
    data['nr_months_ago'] = data.apply(lambda row: nrmonths(row['date'], startdate_y), axis=1)
    data_dynamic_mean = get_dynamic(data, monthly_mean, 'mean', timewindow='monthly')
    data_dynamic_count = get_dynamic(data, monthly_count, 'count', timewindow='monthly')
    data_dynamic_sum = get_dynamic(data, monthly_sum, 'sum', timewindow='monthly')

    # Add dynamic feature with the number of visits
    data_dynamic_visits = one_hot_encode_categoricals(data, ['nr_months_ago'])

    # create datetime features: time between first and last visit
    # and mean time between consecutive visits
    data_dates = add_datetime_features(data)

    # merge
    print("Putting it all together..")
    df = pd.concat([data_categoricals, data_diff, data_bools, data_numericals, data_dynamic_mean,
                    data_dynamic_count, data_dynamic_sum, data_dynamic_visits, data_dates], axis=1)

    # add mean time between visits
    df = add_mean_time_between_visits(df)
    print("Done")

    return df


def aggregate_and_fit_y(y_train, x_train_aggregated):
    """This function aggregates y data and fits it to the fullVisitorId's of the x data"""

    y_train = y_train.groupby(['fullVisitorId'])[['target']].sum().reset_index().copy()
    y_train = pd.merge(pd.DataFrame(x_train_aggregated.index), y_train, on='fullVisitorId', how='left')
    y_train['target'] = y_train['target'].fillna(0)
    return y_train


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
