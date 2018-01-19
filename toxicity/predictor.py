from abc import abstractmethod, ABCMeta
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

TUNING_OUTPUT_DEFAULT = 'tuning.txt'
RANDOM_STATE = 42  # Used for reproducible results


class Predictor(object):
    """
    An abstract class modeling our notion of a predictor.
    Concrete implementations should follow the predictors
    interface
    """
    __metaclass__ = ABCMeta

    def __init__(self, params={}, name=None):
        """
        Base constructor. The input training is expected to be preprocessed and contain
        features extracted for each sample along with the true values for one of the 6 tags.

        :param params: a dictionary of named model parameters
        :param name: Optional model name, used for logging
        """
        self.params = params
        self.name = name

    def __str__(self):
        return self.name

    def set_params(self, params):
        """Override parameters set in the constructor. Dictionary expected"""
        self.params = params

    @abstractmethod
    def fit(self, train_x, train_y):
        """
        A function that fits the predictor to the provided dataset
        """

    @abstractmethod
    def predict(self, test_x):
        """
        Predicts the label for the given input
        :param test_x: a pd.DataFrame of features to be used for predictions
        :return: The predicted labels
        """

    # TODO: Perhaps we can make a wrapper to train the model or separate models across all tags at once.
    def tune(self, train_x, train_y, params, nfolds=3, verbose=3, persist=True, write_to=TUNING_OUTPUT_DEFAULT):
        """
        Exhaustively searches over the grid of parameters for the best combination
        :param train_x Contains the input features
        :param train_y Contains the dependent tag
        :param params: Grid of parameters to be explored
        :param nfolds: Number of folds to be used by cross-validation
        :param verbose: Verbosity level. 0 is silent, higher int prints more stuff
        :param persist: If set to true, will write tuning results to a file
        :param write_to: If persist is set to True, write_to defines the filepath to write to
        :return: tuple of: (Dict of best parameters found, Best score achieved).
        """
        grid = GridSearchCV(self.model, params, scoring="neg_log_loss", cv=nfolds, n_jobs=8, verbose=verbose)
        grid.fit(train_x, train_y)

        if persist:
            # Append results to file.
            with open(write_to, "a") as f:
                f.write("------------------------------------------------\n")
                f.write("Model\t{}\n".format(self.name))
                f.write("Best Score\t{}\nparams: {}\n\n".format(grid.best_score_, grid.best_params_))

        return grid.best_params_, grid.best_score_

    def evaluate(self, x, y, val_size=0.3):
        """
        Evaluate performance based on a train-test split.

        :param x: Input features to be used for fitting
        :param y: Values for the tag to be predicted
        :param val_size: Ratio of the training set to be used as validation
        :return: The average log loss error across all toxic tags
        """
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=val_size, random_state=RANDOM_STATE)

        self.model.fit(train_x, train_y)
        predictions = self.model.predict_proba(val_x)[:, 1]
        return log_loss(val_y, predictions)


class DummyPredictor(Predictor):
    """
    A dummy predictor, marking every comment as clean the median. Used for benchmarking models.
    """
    def __init__(self, params={}, name="Naive"):
        super().__init__(params, name=name)
        self.set_model(None)

    def fit(self, train_x=None, train_y=None):
        """
        A dummy predictor does not require training.
        """
        pass

    def predict(self, test_x):
        """
        Predicts the label for the given input.

        Toxicity tags are rather rare, so a legitimate base predictor would
        predict 0 for every tag.
        :param test_x: a pd.DataFrame of features to be used for predictions
        :return: The predicted labels
        """
        return np.zeros((test_x.shape[0], 1))
