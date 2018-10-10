from abc import abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import timing # noqa

TUNING_OUTPUT_DEFAULT = 'tuning.txt'
RANDOM_STATE = 42  # Used for reproducible results


class Predictor(BaseEstimator, ClassifierMixin):
    """
    An abstract class modeling our notion of a predictor.
    Concrete implementations should follow the predictors
    interface
    """
    name = 'Abstract Predictor'

    def __init__(self, name=name):
        """
        Base constructor. The input training is expected to be preprocessed and contain
        features extracted for each sample along with the true values

        :param params: a dictionary of named model parameters
        :param name: Optional model name, used for logging
        """
        self.name = name

    def __str__(self):
        return self.name

    @abstractmethod
    def fit(self, train_x, train_y):
        """
        A function that fits the predictor to the provided dataset
        """
        self.model.fit(train_x, train_y)

    def score(self, x, y, sample_weight=None):
        """
        A function to calculate the RMSE
        """
        return mean_squared_error(y, self.predict(x)) ** (1/2)

    @abstractmethod
    def predict(self, test_x):
        """
        Predicts the label for the given input
        :param test_x: a pd.DataFrame of features to be used for predictions
        :return: The predicted labels
        """
        return self.model.predict(test_x)

    @abstractmethod
    def predict_proba(self, test_x):
        """
        Predicts the probability of the label for the given input
        :param test_x: a pd.DataFrame of features to be used for predictions
        :return: The predicted probabilities
        """
        return self.model.predict_proba(test_x)

    @timing
    def evaluate(self, x, y, method="split", nfolds=3, val_size=0.3):
        """
        Evaluate performance of the predictor. The default method `CV` is a lot more robust, however it is also a lot slower
        since it goes through `nfolds` iterations. The `split` method is based on a train-test split which makes it a lot faster.

        :param x: Input features to be used for fitting
        :param y: Target values
        :param method: String denoting the evaluation method. Acceptable values are cv for cross validation and split for train-test split
        :param nfolds: Number of folds per tag in case CV is the evaluation method. Ignored otherwise
        :param val_size: Ratio of the training set to be used as validation in case split is the evaluation method. Ignored otherwise
        :return: The average log loss error across all tags
        """
        # print("Using {} evaluation method across all tags...".format(method))
        # if method == 'stratified_CV':
        #    return self._stratified_cv(x, ys, nfolds)

        if method == 'CV':
            scores = cross_val_score(self, x, y, cv=nfolds)
            return np.mean(scores)

        if method == 'split':
            train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=val_size, random_state=RANDOM_STATE)
            self.fit(train_x, train_y)
            predictions = self.predict(val_x)
            return mean_squared_error(val_y, predictions)

        raise ValueError("Method must be either 'stratified_CV', 'CV' or 'split', not {}".format(method))
