import numpy as np
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from predictor import Predictor


class LogisticPredictor(Predictor):
    """
    Adapted to our class design from the kernel:
    https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb
    """
    name = 'Logistic Regression Predictor'

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=None, name=name):

        super().__init__(name)
        n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                        intercept_scaling=intercept_scaling, class_weight=class_weight,
                                        random_state=random_state, solver=solver, max_iter=max_iter,
                                        multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        self.r = None

    def fit(self, train_x, train_y, **params):
        """
        A function that fits the predictor to the provided dataset

        :param train_x Contains the input features
        :param train_y Contains the dependent tag values
        """

        def pr(y_i):
            p = train_x[train_y == y_i].sum(0)
            return (p + 1) / ((train_y == y_i).sum() + 1)

        self.r = np.log(pr(1) / pr(0))
        nb = train_x.multiply(self.r)
        self.model.fit(nb, train_y, **params)

    def predict_proba(self, test_x):
        """
        Predicts the label for the given input

        :param test_x: a (potentially sparse) array of shape: (n_samples, n_features)
        :return: The predicted labels
        """
        m = test_x.multiply(self.r)
        return self.model.predict_proba(m)[:, 1]

    def predict(self, test_x):
        m = test_x.multiply(self.r)
        return self.model.predict(m)


class SVMPredictor(Predictor):
    """
    An linear Predictor based on SVMs.
    """
    name = 'Linear SVM Predictor'

    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                 fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                 max_iter=1000, name=name):

        super().__init__(name=name)
        self.model = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class,
                               fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                               class_weight=class_weight, verbose=verbose, random_state=random_state, max_iter=max_iter)

    def fit(self, train_x, train_y, **params):
        """
        A function that fits the predictor to the provided dataset.

        :param train_x Contains the input features
        :param train_y Contains the dependent tag values
        """
        self.model.fit(train_x, train_y, **params)

    def predict_proba(self, test_x):
        """
        Predicts the probability of the label being 1 for the given input.

        :param test_x: a (potentially sparse) array of shape: (n_samples, n_features)
        :return: The predicted probabilities for each sample
        """
        return self.model.decision_function(test_x)

    def predict(self, test_x):
        """
        Predicts the label for each sample found in the input.

        :param test_x: a (potentially sparse) array of shape: (n_samples, n_features)
        :return: The predicted labels (binary) for each sample
        """
        return self.model.predict(test_x)
