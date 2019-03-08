import numpy as np
import pandas as pd

from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from obspy.signal.trigger import classic_sta_lta


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
    minimum, maximum, mean, median, std: boolean, optional (default: True),
        Whether to include the corresponding feature.
    quantiles: list of floats,
        The quantiles to compute.
    abs_min, abs_max, abs_mean, abs_median, abs_std: boolean, optional (default: True),
        The same features as above, but calculated over the absolute signal.
    abs_quantiles: list of floats,
        The quantiles to compute over the absolute signal.
    mean_abs_delta, mean_rel_delta: boolean, optional (default: True),
        Whether to compute the average change per observation. For 'mean_rel_delta' it is divided
        by the value of the previous observation, which leads to a change proportion.
    max_to_min: boolean, optional (default: True),
        Whether to compute the rate between the absolute maximum and the absolute minimum.
    count_abs_big: boolean, optional (default: True),
        Whether to count the number of absolute values greater than 97.
    count_abs_ext_big: boolean, optional (default: True)
        Whether to count the number of absolute values greater than 500.
    abs_trend: boolean, optional (default: True),
        Whether to calculate the linear trend of the time series.
    mad: boolean, optional (default: True),
        Whether to calculate the mean absolute deviation of the time series.
    skew: boolean, optional (default: True),
        Whether to calculate the skewness of the time series.
    abs_skew: boolean, optional (default: True),
        Whether to calculate the skewness of the absolute values of the time series.
    kurtosis: boolean, optional (default: True),
        Whether to calculate the kurosis of the time series. The kurtosis
        measures the tailedness of a time series
    abs_kurtosis: boolean, optional (default: True),
        Whether to calculate the kurosis of the absolute values of the time series.
        The kurtosis measures the tailedness of a time series
    hilbert: boolean, optional (default: True),
        Whether to calculate the abs mean in hilbert tranformed space.
    hann: boolean, optional (default: True),
        Whether to calculate the abs mean in hann window.
    stalta: boolean, optional (default: True),
        Whether to calculate the short time average over long time average of the
        time series.
    exp_mov_ave: boolean, optional (default: True),
        Whether to calculate the mean of the mean of the exponential moving average.
    window: int or None, optional (default: None),
        If given, calculates the features over subsequences of size 'window'.
    array_length: int, optional (default: 150000),
        The array length to expect. Only needed if window is not None.

    Returns
    -------
    result: np.array,
        The specified features of the given array.

    Notes
    -----
    In order to see which value in the result refers to which feature, see 'self.feature_names'.
    """
    feats = ["minimum", "maximum", "mean", "median", "std", "abs_min", "abs_max", "abs_mean",
             "abs_median", "abs_std", "mean_abs_delta", "mean_rel_delta", "max_to_min", "count_abs_big",
             "count_abs_ext_big", "abs_trend", "mad", "skew", "abs_skew", "kurtosis", "abs_kurtosis",
             "hilbert", "hann"]

    def __init__(self, minimum=True, maximum=True, mean=True, median=True, std=True, quantiles=None,
                 abs_min=True, abs_max=True, abs_mean=True, abs_median=True, abs_std=True, abs_quantiles=None,
                 mean_abs_delta=True, mean_rel_delta=True, max_to_min=True, count_abs_big=True,
                 count_abs_ext_big=True, abs_trend=True, mad=True, skew=True, abs_skew=True, kurtosis=True,
                 abs_kurtosis=True, hilbert=True, hann=True, stalta=True, exp_mov_ave=True, window=None,
                 array_length=150000):

        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.median = median
        self.std = std
        self.abs_min = abs_min
        self.abs_max = abs_max
        self.abs_mean = abs_mean
        self.abs_median = abs_median
        self.abs_std = abs_std
        self.mean_abs_delta = mean_abs_delta
        self.mean_rel_delta = mean_rel_delta
        self.max_to_min = max_to_min
        self.count_abs_big = count_abs_big
        self.count_abs_ext_big = count_abs_ext_big
        self.abs_trend = abs_trend
        self.mad = mad
        self.skew = skew
        self.abs_skew = abs_skew
        self.kurtosis = kurtosis
        self.abs_kurtosis = abs_kurtosis
        self.hilbert = hilbert
        self.hann = hann
        self.stalta = stalta
        self.exp_mov_ave = exp_mov_ave

        if quantiles is None:
            self.quantiles = []
        else:
            self.quantiles = quantiles

        if abs_quantiles is None:
            self.abs_quantiles = []
        else:
            self.abs_quantiles = abs_quantiles

        self.window = window

        if self.stalta is True:
            self.stalta_options = [(50, 1000), (100, 1500), (500, 5000), (1000, 10000), (5000, 15000), (10000, 25000)]
            if self.window:
                self.stalta_options_window = [(50, 1000), (100, 1500), (500, 5000), (1000, 5000)]
        else:
            self.stalta_options = []

        if self.exp_mov_ave is True:
            self.exp_mov_ave_options = [300, 3000, 10000]
            if self.window:
                self.exp_mov_ave_options_window = [300, 1000, 2000]
        else:
            self.exp_mov_ave_options = []

        if self.window is not None:
            self.indicators = np.array(([np.ones(window)*i for i in range(int(np.ceil(array_length/window)))]),
                                       dtype=int).flatten()
            self.indicators = self.indicators[:array_length]
            assert len(self.indicators) == array_length, "Lengths do not match"

        self.feature_names = self._infer_names()
        self.n_features = len(self.feature_names)

    def _infer_names(self):
        """Infer the names of the features that will be calculated."""
        quantile_names = [str(q) + "-quantile" for q in self.quantiles]
        abs_quantile_names = [str(q) + "-abs_quantile" for q in self.abs_quantiles]
        stalta_names = ["all_STALTA-" + str(q[0]) + "-" + str(q[1]) for q in self.STALTA_options]
        exp_mov_ave_names = ["all_exp_mov_ave-" + str(q) for q in self.exp_mov_ave_options]
        if self.window is not None:
            stalta_names_window = ["STALTA-" + str(q[0]) + "-" + str(q[1]) for q in self.STALTA_options_window]
            exp_mov_ave_names_window = ["exp_mov_ave-" + str(q) for q in self.exp_mov_ave_options_window]
        names = np.array(self.feats)[[self.minimum, self.maximum, self.mean, self.median, self.std,
                                      self.abs_min, self.abs_max, self.abs_mean, self.abs_median,
                                      self.abs_std, self.mean_abs_delta, self.mean_rel_delta,
                                      self.max_to_min, self.count_abs_big, self.count_abs_ext_big,
                                      self.abs_trend, self.mad, self.skew, self.abs_skew, self.kurtosis,
                                      self.abs_kurtosis, self.hilbert, self.hann]]
        names = names.tolist() + quantile_names + abs_quantile_names

        if self.window is not None:
            all_names = [str(i) + "_" + name for i in np.unique(self.indicators) for name in names + stalta_names_window + exp_mov_ave_names_window]
            self.result_template_window = np.zeros(int(len(all_names) / len(np.unique(self.indicators))))
            all_names = all_names + ["all_" + name for name in names] + stalta_names + exp_mov_ave_names
            self.result_template = np.zeros(len(names + stalta_names + exp_mov_ave_names))
            return all_names
        else:
            all_names = names + stalta_names + exp_mov_ave_names
            self.result_template = np.zeros(len(all_names))
            return all_names

    def compute(self, arr):
        if self.window is None:
            return self._compute_features(arr)
        else:
            df = pd.DataFrame({"arr": arr, "indicator": self.indicators})
            values = (df.groupby("indicator")["arr"]
                      .apply(lambda x: self._compute_features(x, window=True))
                      .apply(pd.Series)
                      .values
                      .flatten())

            # include values over the whole segment
            overall_values = self._compute_features(arr)

            return np.concatenate([values, overall_values])

    def _compute_features(self, arr, window=False):
        if window:
            result = np.zeros_like(self.result_template_window)
        else:
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
        if self.abs_min:
            result[i] = np.min(np.abs(arr))
            i += 1
        if self.abs_max:
            result[i] = np.max(np.abs(arr))
            i += 1
        if self.abs_mean:
            result[i] = np.mean(np.abs(arr))
            i += 1
        if self.abs_median:
            result[i] = np.median(np.abs(arr))
            i += 1
        if self.abs_std:
            result[i] = np.std(np.abs(arr))
            i += 1
        if self.mean_abs_delta:
            result[i] = np.mean(np.diff(arr))
            i += 1
        if self.mean_rel_delta:
            result[i] = np.mean(np.nonzero((np.diff(arr) / arr[:-1]))[0])
            i += 1
        if self.max_to_min:
            result[i] = np.max(arr) / np.abs(np.min(arr))
            i += 1
        if self.count_abs_big:  # threshold is chosen such that 0.05% of train is above the threshold
            result[i] = len(arr[np.abs(arr) > 97])
            i += 1
        if self.count_abs_ext_big:  # threshold is chosen based on visualization of train
            result[i] = len(arr[np.abs(arr) > 500])
            i += 1
        if self.abs_trend:
            idx = np.array(range(len(arr)))
            lr = LinearRegression()
            lr.fit(idx.reshape(-1, 1), np.abs(arr))
            result[i] = lr.coef_[0]
            i += 1
        if self.mad:  # mean absolute deviation
            result[i] = np.mean(np.abs(arr - np.mean(arr)))
            i += 1
        if self.skew:
            result[i] = stats.skew(arr)
            i += 1
        if self.abs_skew:
            result[i] = stats.skew(np.abs(arr))
            i += 1
        if self.kurtosis:  # measure of tailedness
            result[i] = stats.kurtosis(arr)
            i += 1
        if self.abs_kurtosis:  # measure of tailedness
            result[i] = stats.kurtosis(np.abs(arr))
            i += 1
        if self.hilbert:  # abs mean in hilbert tranformed space
            result[i] = np.mean(np.abs(signal.hilbert(arr)))
            i += 1
        if self.hann:  # mean in hann window
            result[i] = np.mean(signal.convolve(arr, signal.hann(150), mode='same') / np.sum(signal.hann(150)))
            i += 1
        if self.quantiles is not None:
            result[i:i + len(self.quantiles)] = np.quantile(arr, q=self.quantiles)
            i += len(self.quantiles)
        if self.abs_quantiles is not None:
            result[i:i + len(self.abs_quantiles)] = np.quantile(np.abs(arr), q=self.abs_quantiles)
            i += len(self.abs_quantiles)
        if self.stalta:
            if window:
                result[i:i + len(self.stalta_options_window)] = np.array(
                        [np.mean(classic_sta_lta(arr, q[0], q[1])) for q in self.stalta_options_window])
                i += len(self.STALTA_options_window)
            else:
                result[i:i + len(self.stalta_options)] = np.array(
                        [np.mean(classic_sta_lta(arr, q[0], q[1])) for q in self.stalta_options])
                i += len(self.stalta_options)
        if self.exp_mov_ave:
            if window:
                result[i:i + len(self.exp_mov_ave_options_window)] = np.array(
                        [np.mean(pd.Series.ewm(pd.Series(arr), span=q).mean()) for q in self.exp_mov_ave_options_window])
                i += len(self.exp_mov_ave_options_window)
            else:
                result[i:i + len(self.exp_mov_ave_options)] = np.array(
                        [np.mean(pd.Series.ewm(pd.Series(arr), span=q).mean()) for q in self.exp_mov_ave_options])
                i += len(self.exp_mov_ave_options)

        return result


def create_feature_dataset(data, feature_computer, xcol="acoustic_data", ycol="time_to_failure", n_samples=100,
                           stft=False, stft_feature_computer=None):
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
    stft: bool, optional (default: False),
        Whether to calculate the Short Time Fourier Transform.
    stft_feature_computer: FeatureComputer object or None,
        The computer for stft features.

    Returns
    -------
    feature_data: pd.DataFrame,
        A new dataframe of shape (n_samples, number of features) with the new features per sequence.
    """
    if (stft is True) and (stft_feature_computer is None):
        assert feature_computer.window is None, ("If stft is True, feature_computer must have window=None or"
                                                 "a separate stft_feature_computer must be provided.")
        stft_feature_computer = feature_computer

    new_data = pd.DataFrame({feature: np.zeros(n_samples) for feature in feature_computer.feature_names})
    targets = np.zeros(n_samples)
    data_gen = sequence_generator(data, xcol=xcol, ycol=ycol, size=150000)

    if stft:
        new_data_stft = pd.DataFrame({feature + '_stft': np.zeros(n_samples) for feature in stft_feature_computer.feature_names})

    for i in range(n_samples):
        x, y = next(data_gen)
        new_data.iloc[i, :] = feature_computer.compute(x)
        targets[i] = y

        if stft:
            _, _, zxx = signal.stft(x)
            x_stft = np.sum(np.abs(zxx), axis=0)
            new_data_stft.iloc[i, :] = stft_feature_computer.compute(x_stft)

    if stft:
        new_data = pd.concat([new_data, new_data_stft], axis=1)

    new_data[ycol] = targets
    return new_data
