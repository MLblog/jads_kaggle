import numpy as np
import pandas as pd


def sequence_generator(data, xcol="acoustic_data", ycol="time_to_failure", size=150000):
    """Generator that extracts segments of the signal from the data.

    Parameters
    ----------
    data: pd.DataFrame,
        The data with all observations. Must have two columns: one with the measurement
        of the signal and one with the target, i.e., time to the next earthquake.
    xcol: str, optional (default: "acoustic_data"),
        The column referring to the the signal data.
    ycol: str, optional (default: "time_to_failure"),
        The column referring to the target value.
    size: int, optional (default: 150,000),
        The number of observations to include in a single sequence. Should be left at
        its default to generate sequences similar to the test data.

    Returns
    -------
    A generator object that generates tuples like:
    (sequence of 'size' observations of xcol, last corresponding value of ycol).
    """
    while True:
        indices = np.random.randint(0, len(data) - size - 1, 10000)
        for idx in indices:
            y = data[ycol].iloc[idx + size - 1]
            x = data[idx:(idx + size)][xcol].values
            yield x, y


class FeatureComputer():
    """Class that computes features over a given array of observations.

    This is done in a class so that it can be initialized once and can then be used throughout the
    train-validate-test sequence without specifying all the parameters everytime.

    Parameters
    ----------
    minimum, maximum, mean, median, std, quantiles: boolean, optional (default: True),
        Whether to include the corresponding feature.

    Returns
    -------
    result: np.array,
        The specified features of the given array.

    Notes
    -----
    In order to see which value in the result refers to which feature, see 'self.feature_names'.
    """
    feats = ["minimum", "maximum", "mean", "median", "std"]

    def __init__(self, minimum=True, maximum=True, mean=True, median=True, std=True, quantiles=None):
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.median = median
        self.std = std
        if quantiles is None:
            self.quantiles = []
        else:
            self.quantiles = quantiles

        self.feature_names = self._infer_names()
        self.n_features = np.sum([minimum, maximum, mean, median, std, len(self.quantiles)])
        self.result_template = np.zeros(self.n_features)

    def _infer_names(self):
        quantile_names = [str(q) + "-quantile" for q in self.quantiles]
        names = np.array(self.feats)[[self.minimum, self.maximum, self.mean, self.median, self.std]]
        names = names.tolist() + quantile_names
        return names

    def compute(self, arr):
        result = np.zeros_like(self.result_template)
        i = 0
        if self.minimum:
            result[i] = np.min(arr)
            i += 1
        if self.maximum:
            result[i] = np.max(arr)
            i += 1
        if self.mean:
            result[i] = np.mean(arr)
            i += 1
        if self.median:
            result[i] = np.median(arr)
            i += 1
        if self.std:
            result[i] = np.std(arr)
            i += 1
        if self.quantiles is not None:
            result[i:] = np.quantile(arr, q=self.quantiles)
        return result


def create_feature_dataset(data, feature_computer, xcol="acoustic_data", ycol="time_to_failure", n_samples=100):
    """Samples sequences from the data, computes features for each sequence, and stores the result
    in a new dataframe.

    Parameters
    ----------
    data: pd.DataFrame,
        The data with all observations. Must have two columns: one with the measurement
        of the signal and one with the target, i.e., time to the next earthquake.
    feature_computer: FeatureComputer object or similar,
        A class that implements a method '.compute()' that takes an array and returns
        features. It must also have an attribute 'feature_names' that shows the corresponding
        names of the features.
    xcol: str, optional (default: "acoustic_data"),
        The column referring to the the signal data.
    ycol: str, optional (default: "time_to_failure"),
        The column referring to the target value.
    n_samples: int, optional (default: 100),
        The number of sequences to process and return.

    Returns
    -------
    feature_data: pd.DataFrame,
        A new dataframe of shape (n_samples, number of features) with the new features per sequence.
    """
    new_data = pd.DataFrame({feature: np.zeros(n_samples) for feature in feature_computer.feature_names})
    targets = np.zeros(n_samples)
    data_gen = sequence_generator(data, xcol=xcol, ycol=ycol, size=150000)

    for i in range(n_samples):
        x, y = next(data_gen)
        new_data.iloc[i, :] = feature_computer.compute(x)
        targets[i] = y

    new_data[ycol] = targets
    return new_data
