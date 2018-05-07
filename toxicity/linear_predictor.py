import numpy as np
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not imported.")
from predictor import Predictor


class LogisticPredictor(Predictor):
    """
    Adapted to our class design from the kernel:
    https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb
    """
    name = 'Logistic Regression Predictor'

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,  # noqa
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=None, name=name):
        super().__init__(name)
        n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                        intercept_scaling=intercept_scaling, class_weight=class_weight,
                                        random_state=random_state, solver=solver, max_iter=max_iter,
                                        multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

        # Parameters need to be included for cross_validation to work.
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

        # Used for internal representation
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

    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr',  # noqa
                 fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                 max_iter=1000, name=name):
        super().__init__(name=name)
        self.model = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class,
                               fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                               class_weight=class_weight, verbose=verbose, random_state=random_state, max_iter=max_iter)

        # Parameters need to be included for cross_validation to work.
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.penalty = penalty
        self.loss = loss

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


class RFPredictor(Predictor):
    """
    An linear Predictor based on Random Forests.
    """
    name = 'Linear Random Predictor'

    def __init__(self, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 warm_start=False, class_weight=None, name=name):
        super().__init__(name=name)
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                                            max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                            min_impurity_decrease=min_impurity_decrease,
                                            min_impurity_split=min_impurity_split, bootstrap=bootstrap,
                                            oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                                            verbose=verbose, warm_start=warm_start, class_weight=class_weight)

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
        return self.model.predict_proba(test_x)[:, 1]

    def predict(self, test_x):
        """
        Predicts the label for each sample found in the input.

        :param test_x: a (potentially sparse) array of shape: (n_samples, n_features)
        :return: The predicted labels (binary) for each sample
        """
        return self.model.predict(test_x)


class XGBPredictor(Predictor):
    """
    An XGBoost Classifier based on trees.
    """
    name = 'XGBoost Predictor'

    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic',
                 gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None, name=name):
        super().__init__(name=name)
        self.model = XGBClassifier(max_depth=int(max_depth), learning_rate=learning_rate, n_estimators=n_estimators,
                                   silent=silent, objective=objective,
                                   gamma=gamma, min_child_weight=min_child_weight, max_delta_step=max_delta_step,
                                   subsample=subsample, colsample_bytree=colsample_bytree,
                                   colsample_bylevel=colsample_bylevel, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                   scale_pos_weight=scale_pos_weight, base_score=base_score,
                                   seed=seed, missing=missing)

        # Parameters need to be included for cross_validation to work.
        self.max_depth = int(max_depth)
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.seed = seed
        self.missing = missing

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
        return self.model.predict_proba(test_x)[:, 1]

    def predict(self, test_x):
        """
        Predicts the label for each sample found in the input.

        :param test_x: a (potentially sparse) array of shape: (n_samples, n_features)
        :return: The predicted labels (binary) for each sample
        """
        return self.model.predict(test_x)
