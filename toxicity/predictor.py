from abc import abstractmethod, ABCMeta
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


TUNING_OUTPUT_DEFAULT = 'tuning.txt'


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
        features extracted for each sample along with the true values for each of the 6 tags.

        Concrete implementations need to assign a value to self.model
        (for example using self.set_model(LogisticRegression(**params)).
        The assigned value should either be a dictionary from tags to estimators,
        or a single estimator to be used for all tags.

        :param params: a dictionary of named model parameters
        :param name: Optional model name, used for logging
        """
        self.params = params
        self.name = name
        self.tags = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def set_model(self, model):
        for tag in self.tags:
            try:
                # The supplied argument is a dictionary
                self.model[tag] = model[tag]
            except KeyError:
                # The supplied argument is an estimator
                self.model[tag] = model


    def set_params(self, params):
        """Override parameters set in the constructor. Dictionary expected"""
        self.params = params

    @abstractmethod
    def fit(self, train=None):
        """
        A function that fits the predictor to the given dataset.
        """

    @abstractmethod
    def predict(self, test_x):
        """
        Predicts the label for the given input
        :param test_x: a pd.DataFrame of features to be used for predictions
        :return: The predicted labels
        """

    def tune(self, params, nfolds=3, verbose=3, persist=True, write_to=TUNING_OUTPUT_DEFAULT):
        """
        Exhaustively searches over the grid of parameters for the best combination
        :param params: Grid of parameters to be explored
        :param nfolds: Number of folds to be used by cross-validation.
        :param verbose: Verbosity level. 0 is silent, higher int prints more stuff
        :param persist: If set to true, will write tuning results to a file
        :param write_to: If persist is set to True, write_to defines the filepath to write to
        :return: Dict of best parameters found.
        """
        def persist_tuning(score, params):
            """
            Persists a set of parameters as well as their achieved score to a file.
            :param params: Parameters used
            :param score: Score achieved on the test set using params
            :param write_to: If passed, the optimal parameters found will be written to a file
            :return: Void
            """
            with open(write_to, "a") as f:
                f.write("------------------------------------------------\n")
                f.write("Model\t{}\n".format(self.name))
                f.write("Best Score\t{}\nparams: {}\n\n".format(score, params))

        train, _ = self.split()
        train_y = train['target'].values
        train_x = train.drop('target', axis=1)

        grid = GridSearchCV(self.model, params, cv=nfolds, n_jobs=8, verbose=verbose)
        grid.fit(train_x, train_y)

        if persist:
            persist_tuning(grid.best_score_, grid.best_params_)
        return grid.best_params_, grid.best_score_

    def evaluate(self, train, val_size=0.3):
        """
        Evaluate performance based on a train-test split.

        :param val_size: Ratio of the training set to be used as validation
        :return: The average log loss error across all toxic tags
        """
        train, val = train_test_split(train, test_size=val_size, random_state=42)
        train_x = train.drop(self.tags, axis=1)
        val_x = val.drop(self.tags, axis=1)

        loss = []
        for i, tag in enumerate(self.tags):
            train_y = train[tag]
            val_y = val[tag]
            self.model.fit(train_x, train_y)
            predictions = self.model.predict_proba(val_x)[:, 1]
            loss.append(log_loss(val_y, predictions))

        return np.mean(loss)

    def create_submission(self, test_x, write_to='submissions/submission.csv'):
        """
        Creates a submissions file for the given test set

        :param test_x: a pd.DataFrame containing all features but not tags
        :param write_to: A file path where the submission is written
        """
        predictions = self.predict(test_x)
        ids_df = pd.DataFrame({'id': test_x['id']})
        predictions_df = pd.DataFrame(predictions, columns=self.tags)
        submission = pd.concat(ids_df, predictions_df, axis=1)
        submission.to_csv(write_to, index=False)
        print("Submissions created at location " + write_to)


class DummyPredictor(Predictor):
    """
    A dummy predictor, marking every comment as clean the median. Used for benchmarking models.
    """
    def __init__(self, params={}, name="Naive"):
        super().__init__(params, name=name)
        self.set_model(None)

    def fit(self, train=None):
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
        return np.zeros((test_x.shape[0], len(self.tags)))



if __name__ == "__main__":

    print("Reading training data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    print("\nSetting up data for Dummy Predictor ...")
    model = DummyPredictor()

    print("\nEvaluating Dummy Predictor...")
    score = model.evaluate(train)

    print("\n##########")
    print("Log Loss is: {}".format(score))
    print("##########")