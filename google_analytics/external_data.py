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

    # fix some mismatch of countries:
    # The total mismatch is 5% of all data, fix the 8 countries below can reduce the mistach to 0.5%
    gdp.loc[gdp["Country Name"] == "Russian Federation", "Country Name"] = "Russia"
    gdp.loc[gdp["Country Name"] == "Korea, Rep.", "Country Name"] = "South Korea"
    gdp.loc[gdp["Country Name"] == "Hong Kong SAR, China", "Country Name"] = "Hong Kong"
    gdp.loc[gdp["Country Name"] == "Czech Republic", "Country Name"] = "Czechia"
    gdp.loc[gdp["Country Name"] == "Egypt, Arab Rep.", "Country Name"] = "Egypt"
    gdp.loc[gdp["Country Name"] == "Venezuela, RB", "Country Name"] = "Venezuela"
    gdp.loc[gdp["Country Name"] == "Slovak Republic", "Country Name"] = "Slovakia"

    gdp_drop = gdp[["Country Name", "GDP"]].dropna(subset=['GDP']).rename(columns={"Country Name": "country"}).reset_index(drop=True)
    gdp_drop.loc[gdp_drop.index[-1] + 1] = ["Taiwan", 31900]

    return gdp_drop


def get_hdi(path):
    """Process the raw HDI (Human Development Index) originally
    downloaded from `http://hdr.undp.org/en/composite/HDI`
    Returns the mapping between country and HDI.

    Parameters
    ----------
    path : str
        Path to the raw file.

    Returns
    -------
    pd.DataFrame
        Mapping between country and HDI in 2017.
     """

    hdi = pd.read_excel(path, skiprows=6, usecols="B:C", header=None).rename({0: "country", 1: "HDI"}, axis=1)

    # fix some mismatch of countries:
    # The total mismatch is 7% of all data, fix the 8 countries below can reduce the mistach to 0.6%
    hdi.loc[hdi["country"] == "Viet Nam", "country"] = "Vietnam"
    hdi.loc[hdi.index[-1] + 1] = ["Taiwan", 0.907]
    hdi.loc[hdi["country"] == "Russian Federation", "country"] = "Russia"
    hdi.loc[hdi["country"] == "Korea (Republic of)", "country"] = "South Korea"
    hdi.loc[hdi["country"] == "Hong Kong, China (SAR)", "country"] = "Hong Kong"
    hdi.loc[hdi["country"] == "Venezuela (Bolivarian Republic of)", "country"] = "Venezuela"
    hdi.loc[hdi["country"] == "Bosnia and Herzegovina", "country"] = "Bosnia & Herzegovina"
    hdi.loc[hdi["country"] == "The former Yugoslav Republic of Macedonia", "country"] = "Macedonia (FYROM)"

    return hdi


def add_external_data(df, external_data, input_type="raw", drop=True):
    """Adds the external data to the dataset. Assuming that both passed dataframes have a column named `country`.

    Parameters
    ----------
    df : pd.DataFrame
        Training of test dataframe including the `country` column.
    external_data : pd.DataFrame
        Mapping between `country` and `external_data`
    input_type : {"raw", "aggregated"}
        Whether the operation should run on the raw, or the aggregated dataset.
    drop : bool
        Whether the old country columns should be droped.

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

    if input_type == "aggregated":
        country_cols = [col for col in df.columns if col.startswith("country") and col != "country"]

        def inverse_ohe(row):
            for c in country_cols:
                if row[c] == 1:
                    return c.split("_")[1]

        df["country"] = df.apply(inverse_ohe, axis=1)
        if drop:
            df = df.drop(country_cols, axis=1)

    elif input_type != "raw":
        msg = "Only {} and {} are supported. \n" + \
              "\tThe former assumes the original form where only the JSON has been flattened.\n" + \
              "\tThe latter assumes that OHE has already occured on top."
        raise ValueError(msg)

    df["country"] = df["country"].fillna("Unknown").apply(stringify)
    result = df.merge(external_data, on="country", how='left')
    if drop:
        result.drop("country", axis=1, inplace=True)

    return result
