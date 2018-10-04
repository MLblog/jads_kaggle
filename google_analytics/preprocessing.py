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
