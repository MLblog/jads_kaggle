import pandas as pd
import numpy as np


def get_gdp(path):
    """Process the raw GDP per capita data originally
    downloaded from `https://data.worldbank.org/indicator/NY.GDP.PCAP.CD`

    Returns the mapping between country and GDP.

    Examples
    --------
    >>> gdp = get_gdp("../data/raw_gdp.csv")
    >>> gdp.head()
        Country Name 	GDP
    0 	Aruba 	       25324.0
    1 	Afghanistan    585.0
    2 	Angola 	       4170.0
    3 	Albania 	   4537.0
    4 	Andorra 	   39146.0

    Parameters
    ----------
    path : str
        Path to the raw file.

    Returns
    -------
    pd.DataFrame
        Mapping between country and latest known GDP.

    """
    gdp = pd.read_csv(path, skiprows=4)

    def forward_fill(row, year=2017):
        """Get the data for the last available year. """
        if year < 1960:
            # All values are missing.
            return np.nan

        value = row[str(year)]
        if not np.isnan(value):
            return int(value)

        # Recursively look at the previous year
        return forward_fill(row, year - 1)

    gdp["GDP"] = gdp.apply(forward_fill, axis=1)
    return gdp[["Country Name", "GDP"]].dropna(subset=['GDP']).rename(columns={"Country Name": "country"})


def add_gdp(df, gdp):
    """Adds the `GDP` to the dataset. Assuming that both passed dataframes have a column named `country`.

    Parameters
    ----------
    df : pd.DataFrame
        Training of test dataframe including the `country` column.

    gdp : pd.DataFrame
        Mapping between `country` and `GDP`

    Returns
    -------
    pd.DataFrame
        The passed `df` with a new column corresponding to the mapped GDP.

    """

    def stringify(maybe_string):
        # Handles Unicode country names like "Côte d’Ivoire" , "Réunion" etc, as well as countries only existing
        # in one of the two dataframes.
        try:
            return str(maybe_string)
        except UnicodeEncodeError:
            return "Unknown"

    df["country"] = df["country"].fillna("Unknown").apply(stringify)
    return df.merge(gdp, on="country", how='left')
