import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm


class Caller:
    """Class to read the data from memory.
    The aim of this class is to read data from the
    memory and to discart it after being unsed. Thus, this is
    an option to deal with memory issues.

    Parameters
    ----------
    size: int
        size of the interval to be stored in
        hard disk
    save_dir: str
        name of the folder where the chunks
        of data will be stored

    Arguments
    ---------
    index_list: list
        names of all the files saved at memory
    intervals: pd.Intervals
        intervals of the data chunks
    """
    def __init__(self, size, save_dir):
        self.size = size
        self.save_dir = save_dir
        self.index_list = None
        self.intervals = None

    def save_data(self, data):
        """ Funtion to save the data partitions

        Parameters
        ----------
        data: pd.DataFrame
            complete data set with the earthquake
            information
        """
        self.index_list = np.linspace(0, data.shape[0], math.ceil(data.shape[0]/self.size)).astype(int)
        range_save = [(x, y-1) for x, y in zip(self.index_list, self.index_list[1:])]
        self.intervals = pd.IntervalIndex.from_arrays(self.index_list[0:len(self.index_list)-1], self.index_list[1::], closed='left')
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
            print("Directory ", self.save_dir,  " Created ")

        for i_0, i_f in tqdm(range_save):
            name = os.path.join(self.save_dir, "{}.pkl".format(i_0))
            data.loc[i_0:i_f, :].to_pickle(name)

    def get_intervals(self, i_init, window_size=150000):
        """Funtion to get data from memory.

        Parameters
        ----------
        i_init: int
            index of the row position where the
            data window will be taken from
        window_size: int, (default 150000)
            number of rows to be taken from the data
        """
        logical_gate = [(i_init in inter) or (i_init + window_size in inter) for inter in self.intervals]
        get_intervals = [x.left for x in self.intervals[logical_gate]]
        if len(get_intervals) == 1:
            get_files = get_intervals
        else:
            logical_gate_intervals = (self.index_list >= get_intervals[0]) & (self.index_list <= get_intervals[-1:])
            get_files = self.index_list[logical_gate_intervals]

        signal = pd.DataFrame([])
        for file in get_files:
            new_file = pd.read_pickle(os.path.join(self.save_dir, "{}.pkl".format(file)))
            signal = pd.concat([signal, new_file], axis=0)
        return signal.loc[i_init: i_init + window_size - 1, :]


def create_feature_dataset_source(caller_cl, feature_computer, xcol="acoustic_data", ycol="time_to_failure",
                                  stft=False, stft_feature_computer=None, events_id=(245829584, 307838916),
                                  window_size=150000, step=100):
    """Samples sequences from the data, computes features for each sequence, and stores the result
    in a new dataframe. This function is a modification from the 'create feature' at the engineering
    script.

    Parameters
    ----------
    caller_sm: caller class
    feature_computer: FeatureComputer object or similar,
        A class that implements a method '.compute()' that takes an array and returns
        features. It must also have an attribute 'feature_names' that shows the corresponding
        names of the features.
    xcol: str, optional (default: "acoustic_data"),
        The column referring to the the signal data.
    ycol: str, optional (default: "time_to_failure"),
        The column referring to the target value.
    stft: bool, optional (default: False),
        Whether to calculate the Short Time Fourier Transform.
    stft_feature_computer: FeatureComputer object or None,
        The computer for stft features.
    events_id: tuple
        index value that define the range of values that we are
        going to take the data from.
    window_size: int
        number of observations to take to get
        information about y
    step: int
        number of observations to skip between ys

    Returns
    -------
    feature_data: pd.DataFrame,
        A new dataframe of shape (number_intervals, number of features) with the new features per sequence.
        The index corresponds to the y position.
    """
    number_intervals = math.floor((events_id[1] - events_id[0])/step) + 1
    indices = [events_id[0] + step * i for i in range(number_intervals)]
    if (events_id[1] - events_id[0]) % step != 0:
        indices.append(events_id[1])

    if (stft is True) and (stft_feature_computer is None):
        assert feature_computer.window is None, ("If stft is True, feature_computer must have window=None or"
                                                 "a separate stft_feature_computer must be provided.")
        stft_feature_computer = feature_computer

    new_data = pd.DataFrame({feature: np.zeros(number_intervals) for feature in feature_computer.feature_names})
    targets = np.zeros(number_intervals)
    target_id = np.zeros(number_intervals, dtype=int)
    if stft:
        new_data_stft = pd.DataFrame({feature + '_stft': np.zeros(number_intervals) for feature in stft_feature_computer.feature_names})

    for i, idx in enumerate(tqdm(indices)):
        data = caller_cl.get_intervals(i_init=idx,
                                       window_size=window_size)

        y = data[ycol].values[-1:]
        x = data[xcol].values
        new_data.iloc[i, :] = feature_computer.compute(x)
        targets[i] = y
        target_id[i] = data.index[-1]

        if stft:
            _, _, zxx = signal.stft(x)
            x_stft = np.sum(np.abs(zxx), axis=0)
            new_data_stft.iloc[i, :] = stft_feature_computer.compute(x_stft)

    if stft:
        new_data = pd.concat([new_data, new_data_stft], axis=1)

    new_data[ycol] = targets
    new_data.index = target_id
    return new_data


def create_signal_dataset(caller_cl, xcol="acoustic_data", ycol="time_to_failure",
                          events_id=(245829584, 307838916), window_size=150000, step=100):
    """Funtion to get the signal values before the event happen.
    caller_sm: caller class
    xcol: str, optional (default: "acoustic_data"),
        The column referring to the the signal data.
    ycol: str, optional (default: "time_to_failure"),
        The column referring to the target value.
        events_id: tuple
    index value that define the range of values that we are
        going to take the data from.
    window_size: int
        number of observations to take to get
        information about y
    step: int
        number of observations to skip between ys

    Returns
    -------
    feature_data: pd.DataFrame,
        A new dataframe of shape (number_intervals, window_size).
        The index corresponds to the y position.
    """
    number_intervals = math.floor((events_id[1] - events_id[0])/step) + 1
    indices = [events_id[0] + step * i for i in range(number_intervals)]
    if (events_id[1] - events_id[0]) % step != 0:
        indices.append(events_id[1])
    new_data = np.zeros((len(indices), window_size))
    targets = np.zeros(number_intervals)
    target_id = np.zeros(number_intervals, dtype=int)

    for i, idx in enumerate(tqdm(indices)):
        data = caller_cl.get_intervals(i_init=idx,
                                       window_size=window_size)
        y = data[ycol].values[-1:]
        x = data[xcol].values
        new_data[i, :] = x
        targets[i] = y
        target_id[i] = data.index[-1]

    new_data = pd.DataFrame(new_data)
    new_data[ycol] = targets
    new_data.index = target_id
    return new_data


def check_distributions_train_test(x_train, x_test, n_col=4, alpha=0.05, **kwars):
    """Check if the Ys on the train and test sets
    are from the same distributions using Kolmogorov-Smirnov test.
    
    Paramaters
    ----------
    x_train: pd.DataFrame
        The data with the features of the train set
    x_test: pd.DataFrame
        The data with the features of the test set 
    n_col = int (default 4)
        number of columns for the grid plot
    alpha: float (default 0.05)
        related with the level of confidence for the
        statistical test. Confidence = 1 - alpha
    **kwars: dict
        parameters for the hist
    """

    def twoSampleKS(x1, x2):
        """Compute the Kolmogorov-Smirnov statistic on 2 samples.
        More information at:

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

        Parameters
        ----------
        X1, X2: 1-D array
            two arrays of sample observations assumed to be drawn from a continuous 
            distribution, sample sizes can be different

        """
        _, p_value = stats.ks_2samp(x1, x2)
        if p_value <= alpha:
            return 'Not Same Dist'
        else:
            return 'Same Dist'

    names = list(x_train.columns)
    n_rows = math.ceil(len(names)/n_col) #number of rows
    pos = 0
    fig, axes = plt.subplots(n_rows, n_col, figsize=(15,15))
    plt.subplots_adjust(hspace=0.4)
    for i in range(n_rows):
        for j in range(n_col):
            if pos >= len(names):
                axes[i, j].spines['top'].set_visible(False)
                axes[i, j].spines['right'].set_visible(False)
                axes[i, j].spines['left'].set_visible(False)
                axes[i, j].spines['bottom'].set_visible(False)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            else:
                test = x_test.values[:, pos]
                train = x_train.values[:, pos]
                hypothesis = twoSampleKS(test, test)
                title = names[pos] + ": " + hypothesis
                if pos==0:
                    freq_p, _, _ = axes[i, j].hist(train, alpha=0.5, color = '#ca0020', label='train', **kwars)
                    freq_y, _, _ = axes[i, j].hist(test, alpha=0.5, color='#2c7bb6', label='test', **kwars)
                else:
                    freq_p, _, _ = axes[i, j].hist(train, alpha=0.5, label=None, color='#ca0020', **kwars) 
                    freq_y, _, _ = axes[i, j].hist(test, alpha=0.5, label=None, color='#2c7bb6', **kwars)

                axes[i, j].set_title(title, fontsize=10)
                axes[i, j].spines['top'].set_visible(False)
                axes[i, j].spines['right'].set_visible(False)
                pos += 1

    fig.legend(loc = 'center right')
    fig.suptitle("Test distributions (" + r'$\alpha$=' + '{})'.format(alpha), fontsize=15, y=0.92)
    fig.text(0.08, 0.5, 'Frequency (normalized)', va='center', rotation='vertical', fontsize=15)
