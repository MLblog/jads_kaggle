import numpy as np
import pandas as pd
import os
import gc

from scipy import signal

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from common.utils import progress
from earthquakes.engineering import create_feature_dataset


def train_and_predict(train_x, train_y, val_x, model_cls, return_model=False, **model_params):
    """Convenience function that trains a model on one dataset and predicts
    on another, given a model class and a set of parameters.

    Parameters
    ----------
    train_x, val_x: pd.DataFrame,
        The datasets to train and predict on respectively.
    train_y: array-like,
        The true labels/targets of train_x.
    model_cls: uninitialized predictor class,
        Must implement the '.predict' and '.fit' methods. Every Scikit-Learn
        predictor suffices.
    **model_params: key-value pairs,
        Any parameters that should be used to initialize the model_cls.

    Returns
    -------
    predictions: array,
        The predicted labels for val_x.
    """
    model = model_cls(**model_params)
    model.fit(train_x, train_y)
    predictions = model.predict(val_x)
    if return_model:
        return predictions, model
    else:
        return predictions


def cv_with_feature_computer(data, model_cls, feature_computer, ycol="time_to_failure",
                             n_splits=5, train_samples=1000, val_samples=500, predict_test=False,
                             stft=False, stft_feature_computer=None, data_dir="../data/test", **model_params):
    """Perform custom cross validation using randomly sampled sequences of observations.

    Parameters
    ----------
    data: pd.DataFrame,
        The data with all observations. Must have two columns: one with the measurement
        of the signal and one with the target, i.e., time to the next earthquake.
    model_cls: uninitialized predictor class,
        Must implement the '.predict' and '.fit' methods. Every Scikit-Learn
        predictor suffices.
    feature_computer: FeatureComputer object or similar,
        A class that implements a method '.compute()' that takes an array and returns
        features. It must also have an attribute 'feature_names' that shows the corresponding
        names of the features.
    ycol: str, optional (default: "time_to_failure"),
        The column referring to the target value.
    n_splits: int, optional (default: 5)
        The number of folds in cross validation.
    train_samples: int, optional (default: 1000),
        The number of sequences to sample for training.
    val_samples: int, optional (default: 500),
        The number of sequences to sample for validation.
    predict_test: boolean, optional (default: False),
        If True, predicts on the test data at every fold and returns (together with cv scores)
        a dataframe with predictions on the test data.
    stft: boolean, optional (default: False),
        If true, predicts the Compute the Short Time Fourier Transform.
    stft_feature_computer: FeatureComputer object or None,
        The computer for stft features.
    data_dir: str, optional (default: "../data")
        The path to the main folder with the for this competition data. Note that test data is
        in several files, which are assumed to be in a subfolder called 'test' inside the data_dir.
        A file 'sample_submission.csv' is assumed to be directly in data_dir. This parameter
        is ignored if predict_test=False.
    **model_params: key-value pairs,
        Any parameters to pass to the predictor model_cls upon initialization.

    Returns
    -------
    Either a list of validation scores (if predict_test=False) or a tuple of
    (list of validation scores, DataFrame with test predictions).
    """
    splitter = KFold(n_splits=n_splits, shuffle=False)

    scores = []
    for i, (train_index, val_index) in enumerate(splitter.split(data)):
        progress("Starting cross-validation fold {}.".format(i))

        # split the data according to the indices
        progress("Splitting data in train and validation sets.")
        cols = data.columns
        train = pd.DataFrame(data.values[train_index], columns=cols)
        val = pd.DataFrame(data.values[val_index], columns=cols)

        # sample random sequences for training
        progress("Sampling {} sequences from training data.".format(train_samples))
        train_features = create_feature_dataset(train, feature_computer, n_samples=train_samples,
                                                stft=stft, stft_feature_computer=stft_feature_computer)
        y_train = train_features[ycol]
        x_train = train_features.drop(ycol, axis=1)
        progress("Train set sampled.")

        # sample random sequences for validation
        progress("Sampling {} sequences from validation data.".format(val_samples))
        val_features = create_feature_dataset(val, feature_computer, n_samples=val_samples,
                                              stft=stft, stft_feature_computer=stft_feature_computer)
        y_val = val_features[ycol]
        x_val = val_features.drop(ycol, axis=1)
        progress("Validation set sampled.")

        # train and predict validation set
        progress("Start training and predicting.")
        y_val_hat, model = train_and_predict(x_train, y_train, x_val, model_cls, return_model=True, **model_params)
        progress("Predictions on validation set made.")

        # evaluate using mean absolute error for this competition
        score = mean_absolute_error(y_val, y_val_hat)
        scores.append(score)
        progress("Validation score: {}.".format(score))

        # predict on test set if specified
        if predict_test:
            if i == 0:
                test_predictions = predict_on_test(model, feature_computer, data_dir=data_dir, ycol=ycol,
                                                   stft=stft, stft_feature_computer=stft_feature_computer)
            else:
                new_predictions = predict_on_test(model, feature_computer, data_dir=data_dir, ycol=ycol,
                                                  stft=stft, stft_feature_computer=stft_feature_computer)
                test_predictions[ycol + "_{}".format(i)] = new_predictions[ycol].copy()
        progress("Predictions on test set made.")

        # clear up memory
        del train, val, train_features, y_train, x_train, val_features, x_val, y_val, model
        gc.collect()

    if predict_test:
        return scores, test_predictions
    else:
        return scores


def predict_on_test(model, feature_computer, ycol="time_to_failure", stft=True, stft_feature_computer=None,
                    data_dir="../data", ):
    """Load the test data, compute features on every segment, and predict the target.

    Parameters
    ----------
    model: a fitted predictor,
        Must implement the 'predict' method to predict on the test sequences and
        must already be fitted/trained.
    feature_computer: FeatureComputer object or similar,
        A class that implements a method '.compute()' that takes an array and returns
        features. It must also have an attribute 'feature_names' that shows the corresponding
        names of the features. The same instance as was used during training.
    ycol: str, optional (default: "time_to_failure"),
        The column referring to the target value.
    data_dir: str, optional (default: "../data")
        The path to the main folder with the for this competition data. Note that test data is
        in several files, which are assumed to be in a subfolder called 'test' inside the data_dir.
        A file 'sample_submission.csv' is assumed to be directly in data_dir.
    stft: bool, optional (default: False),
        Whether to calculate the Short Time Fourier Transform.
    stft_feature_computer: FeatureComputer object or None,
        The computer for stft features.

    Returns
    -------
    submission: pd.DataFrame,
        The predictions in the right format for submission.
    """
    # take the segment ids from the sample submission file
    sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"), index_col="seg_id")
    x_test = pd.DataFrame(columns=feature_computer.feature_names, dtype=np.float64, index=sample_submission.index)
    if stft:
        x_test_stft= pd.DataFrame(columns=[x + "_stft" for x in stft_feature_computer.feature_names],  # noqa
                                  dtype=np.float64,
                                  index=sample_submission.index)

    # load and predict segments one by one
    for i, seg_id in enumerate(x_test.index):
        progress("Loading and computing features for segment {}/{}.".format(i + 1, len(x_test)),
                 same_line=True, newline_end=(i + 1 == len(x_test)))

        segment = pd.read_csv(os.path.join(data_dir, "test", seg_id + ".csv"))
        x_test.loc[seg_id, :] = feature_computer.compute(segment["acoustic_data"].values)
        if stft:
            # _, _, zxx = signal.stft([item for sublist in segment["acoustic_data"].values for item in sublist])
            _, _, zxx = signal.stft(segment["acoustic_data"].values)
            x_stft = np.sum(np.abs(zxx), axis=0)
            x_test_stft.loc[seg_id, :] = stft_feature_computer.compute(x_stft)

    if stft:
        x_test = pd.concat([x_test, x_test_stft], axis=1)

    sample_submission[ycol] = model.predict(x_test)
    progress("Predictions made.")
    return sample_submission.reset_index()
