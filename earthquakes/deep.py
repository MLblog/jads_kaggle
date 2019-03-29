import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

from earthquakes.engineering import get_cycle

class Scaler():
    """Class that takes care of scaling the data in various ways.

    The following methods are available:
    - 'log': take the natural logarithm of each value + a constant to prevent negative inputs.
    - 'standard': standardizing by subtracting the mean and dividing by standard deviation.
    - 'abslog': take the log of the absolute value (+ a small value to prevent log(0)).
    - 'minmax': places the value in [0, 1] where 0 represents the min and 1 the max.
    - 'absmax': divides the value by the absolute maximum, so that values are in [-1, 1].
    - 'value': divide by a custom value.

    Parameters
    ----------
    method: str
        The scaling method. One of ['log', 'standard', 'abslog', 'minmax', 'absmax', 'value'].
    epsilon: float, optional, default=1e-6
        Small value that is added to logarithm inputs to prevent taken log(0).
    add_constant: int, optional, default=5516
        A constant that is added to the whole signal when using 'log' as scaling
        method in order to prevent log(x), x <= 0. Default is set to the minimum
        of the training data - 1.

    Notes
    -----
    For the following scaling methods, you need to call `fit()` before you can use `scale`:
    ['standard', 'minmax', 'absmax'].
    """
    def __init__(self, method="log", epsilon=1e-6, add_constant=5516, value=100):
        self.method = method
        self.epsilon = epsilon
        self.C = add_constant
        self.value = value

        self.fitted = False

    def fit(self, arr):
        self.mu = np.mean(arr)
        self.sigma = np.std(arr)
        self.min = np.min(arr)
        self.max = np.max(arr)
        self.absmax = np.max(np.abs(arr))

        self.fitted = True

    def scale(self, arr):
        if self.method == "log":
            return np.log(np.asarray(arr) + self.C)
        if self.method == "standard":
            assert self.fitted, "Fit first if using standard scaling"
            return (np.asarray(arr) - self.mu) / self.sigma
        if self.method == "abslog":
            return np.log(np.abs(arr) + self.epsilon)
        if self.method == "minmax":
            assert self.fitted, "Fit first if using minmax scaling"
            return (np.asarray(arr) - self.min) / (self.max - self.min)
        if self.method == 'absmax':
            assert self.fitted, "Fit first if using absmax scaling"
            return np.asarray(arr) / self.absmax
        if self.method == "value":
            return np.asarray(arr) / self.value


def train_on_cycles(model, epochs=5, batch_size=32, cycle_nrs=None, scaler=None,
                    sequence_length=150000, xcol="acoustic_data", ycol="time_to_failure",
                    data_dir="../data"):
    """Train a Keras model on specific earthquake cycles.

    Parameters
    ----------
    model: keras.models.Model
        The model to train.
    epochs: int, optional, default=5
        The number of epochs to train.
    batch_size: int, optional, default=32
        The number of samples in a batch to use in Gradient Descent.
    cycle_nrs: list of ints
        The cycle numbers you want to train the model on.
    scaler: earthquakes.deep.Scaler object
        Scaler instance to use to scale every cycle. Must be initialized and fitted
        (if the scaling method requires so).
    sequence_length: int, optional, default=150000
        The length of a signal sequence. This should probably be left at its default.
    xcol, ycol: str, optional
        The column names of the signal (xcol) and target (ycol). Defaults to
        xcol="acoustic_data", ycol="time_to_failure".
    data_dir: str, optional, default="../data"
        The directory that holds the cycle data.
    """
    if cycle_nrs is None:
        cycle_nrs = range(17)

    # train for 'epochs' epochs on every cycle in cycle_nrs
    for cycle_nr in cycle_nrs:
        # load cycle
        train_x, train_y = get_cycle(cycle_nr, xcol=xcol, ycol=ycol, data_dir=data_dir)
        # scale signal data
        if scaler is not None:
            train_x = scaler.scale(train_x)
        # create generator
        data_gen = TimeseriesGenerator(train_x, train_y, length=sequence_length, batch_size=batch_size, shuffle=True)
        # train
        model.fit_generator(
            data_gen,
            steps_per_epoch=len(train_x)/sequence_length,
            epochs=epochs,
            use_multiprocessing=False
        )

    return model


def evaluate_on_cycles(model, cycle_nrs=None, scaler=None, sequence_length=150000,
                       xcol="acoustic_data", ycol="time_to_failure", data_dir="../data"):
    """Evaluate a model on certain earthquake cycles.

    Parameters
    ----------
    model: a trained Keras.Model
        Must implement the `predict`.
    cycle_nrs: list of ints
        The cycle numbers you want to evaluate the model on.
    scaler: earthquakes.deep.Scaler object
        Scaler instance to use to scale every cycle. Must be initialized and fitted
        (if the scaling method requires so).
    sequence_length: int, optional, default=150000
        The length of a signal sequence. This should probably be left at its default.
    xcol, ycol: str, optional
        The column names of the signal (xcol) and target (ycol). Defaults to
        xcol="acoustic_data", ycol="time_to_failure".
    data_dir: str, optional, default="../data"
        The directory that holds the cycle data.
    """
    losses, weights = [], []
    for nr in cycle_nrs:
        x, y = get_cycle(nr, xcol=xcol, ycol=ycol, data_dir=data_dir)
        x = x.reshape((len(x), 1))
        y = y.reshape((len(y), 1))
        x = scaler.scale(x)

        data_gen = TimeseriesGenerator(x, y, length=sequence_length, batch_size=128, shuffle=True)
        progress("Evaluating cycle {}..".format(nr))
        loss = model.evaluate_generator(data_gen, steps=len(x)/sequence_length)
        losses.append(loss)
        weights.append(len(x))

    weighted_loss = np.dot(losses, weights) / np.sum(weights)
    print("Weighted loss over cycles: {}".format(weighted_loss))
    return weighted_loss, losses, weights


class KFoldCycles():
    """K-Fold splitter for earthquake cycles.

    Note that folds are not of equal size (not even in the number of cycles), since we
    have 17 cycles, which is a prime number.

    Parameters
    ----------
    k_folds: int, optional, default=4
        Number of folds.
    shuffle: bool, optional, default=True
        Whether to shuffle the cycle numbers before folding.
    """
    def __init__(self, k_folds=4, shuffle=True):
        self.k_folds = int(k_folds)
        self.shuffle = shuffle

    def split(self, n_cycles=17):
        """Create train-validation splits.

        Parameters
        ----------
        n_cycles: int, optional, default=17
            The number of cycles that are available in total.
        """
        cycles = np.arange(n_cycles)
        if self.shuffle:
            np.random.shuffle(cycles)
        cycles_per_fold = n_cycles / self.k_folds  # probably not an integer
        folds = [cycles[int(np.floor(i * cycles_per_fold)):int(np.floor((i + 1) * cycles_per_fold))] for i in range(self.k_folds)]
        for k in range(self.k_folds):
            yield np.concatenate(folds[0:k] + folds[(k + 1):]), folds[k]
