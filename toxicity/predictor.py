from abc import abstractmethod
import numpy as np
from collections import Counter
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin

from utils import timing, TAGS

TUNING_OUTPUT_DEFAULT = 'tuning.txt'
RANDOM_STATE = 42  # Used for reproducible results


class Predictor(BaseEstimator, ClassifierMixin):
    """
    An abstract class modeling our notion of a predictor.
    Concrete implementations should follow the predictors
    interface
    """

    def __init__(self, name=None):
        """
        Base constructor. The input training is expected to be preprocessed and contain
        features extracted for each sample along with the true values for one of the 6 tags.

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

    def score(self, x, y, sample_weight=None):
        return log_loss(y, self.predict_proba(x))

    @abstractmethod
    def predict(self, test_x):
        """
        Predicts the label for the given input
        :param test_x: a pd.DataFrame of features to be used for predictions
        :return: The predicted labels
        """

    @abstractmethod
    def predict_proba(self, test_x):
        """
        Predicts the probability of the label for the given input
        :param test_x: a pd.DataFrame of features to be used for predictions
        :return: The predicted probabilities
        """

    # TODO: This needs to be improved because:
    #       1. It returns a negative score which is unexpected
    #       2. It only works for one tag. We need to make it compute the average score for all tags.
    def tune(self, train_x, train_y, params, nfolds=3, verbose=5, persist=True, write_to=TUNING_OUTPUT_DEFAULT):
        """
        Exhaustively searches over the grid of parameters for the best combination by minimizing the log loss.

        :param train_x Contains the input features
        :param train_y Contains the dependent tag
        :param params: Grid of parameters to be explored
        :param nfolds: Number of folds to be used by cross-validation
        :param verbose: Verbosity level. 0 is silent, higher int prints more stuff
        :param persist: If set to true, will write tuning results to a file
        :param write_to: If persist is set to True, write_to defines the filepath to write to
        :return: tuple of: (Dict of best parameters found, Best score achieved).
        """
        scoring = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
        grid = GridSearchCV(self, params, scoring=scoring, cv=nfolds, n_jobs=8, verbose=verbose)
        grid.fit(train_x, train_y)

        if persist:
            # Append results to file.
            with open(write_to, "a") as f:
                f.write("------------------------------------------------\n")
                f.write("Model\t{}\n".format(self.name))
                f.write("Best Score\t{}\nparams: {}\n\n".format(grid.best_score_, grid.best_params_))

        return grid.best_params_, grid.best_score_

    def _stratified_cv(self, x, ys, nfolds):
        # In order to use stratified CV we transform the multi-label problem into a single label, multi-class one.
        # This is achieved by converting each label set to a single label, using bin -> dec conversion.
        def convert_label(label):
            """
            Translates a set of labels into a single label using binary to decimal conversion
            """
            mul = [2 ** i for i in range(0, len(TAGS))]
            return np.dot(mul, label)

        def delete(arr, indices):
            """
            Delete rows from a sparse matrix by index
            """
            mask = np.ones(arr.shape[0], dtype=bool)
            mask[indices] = False
            return arr[mask]

        # Multi label to single label
        ys = np.array([ys[i] for i in TAGS]).T
        y = np.apply_along_axis(convert_label, 1, ys)

        # Remove rare labels
        c = Counter(y)
        bad_indices = []
        for i, label in enumerate(y):
            if c[label] < 5:
                bad_indices.append(i)
        x = delete(x, bad_indices)
        y = delete(y, bad_indices)

        splitter = StratifiedShuffleSplit(n_splits=nfolds, random_state=RANDOM_STATE)
        scores = []
        for train_index, val_index in splitter.split(x, y):
            train_x, val_x = x[train_index], x[val_index]
            train_ys, val_ys = ys[train_index, :], ys[val_index, :]

            losses = []
            for tag in range(0, len(TAGS)):
                self.fit(train_x, train_ys[:, tag])
                predictions = self.predict_proba(val_x)
                losses.append(log_loss(val_ys[:, tag], predictions))
            scores.append(np.mean(losses))

        return np.mean(scores)

    @timing
    def evaluate(self, x, ys, method="CV", nfolds=3, val_size=0.3):
        """
        Evaluate performance of the predictor. The default method `CV` is a lot more robust, however it is also a lot slower
        since it goes through `nfolds * len(TAGS)` iterations. The `split` method is based on a train-test split which makes it a lot faster.

        :param x: Input features to be used for fitting
        :param ys: Dictionary mapping a tag with its true labels
        :param method: String denoting the evaluation method. Acceptable values are cv for cross validation and split for train-test split
        :param nfolds: Number of folds per tag in case CV is the evaluation method. Ignored otherwise
        :param val_size: Ratio of the training set to be used as validation in case split is the evaluation method. Ignored otherwise
        :return: The average log loss error across all tags
        """
        print("Using {} evaluation method across all tags...".format(method))
        if method == 'stratified_CV':
            return self._stratified_cv(x, ys, nfolds)

        losses = []
        if method == 'CV':
            for tag in TAGS:
                print("Evaluating tag {}".format(tag))
                scores = cross_val_score(self, x, ys[tag], cv=nfolds)
                losses.append(np.mean(scores))
            return np.mean(losses)

        if method == 'split':
            for tag in TAGS:
                train_x, val_x, train_y, val_y = train_test_split(x, ys[tag], test_size=val_size, random_state=RANDOM_STATE)
                self.fit(train_x, train_y)
                predictions = self.predict_proba(val_x)
                losses.append(log_loss(val_y, predictions))
            return np.mean(losses)

        raise ValueError("Method must be either 'stratified_CV', 'CV' or 'split', not {}".format(method))


