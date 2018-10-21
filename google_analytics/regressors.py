import multiprocessing
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
# try:
#     from xgboost import XGBClassifier
# except ImportError:
#     print("XGBoost not imported.")
from predictor import Predictor


class RandomForestPredictor(Predictor):
    name = 'RandomForest Predictor'

    def __init__(self, n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, verbose=0, warm_start=False, name=name):
        n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        super().__init__(name)
        self.model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                           min_weight_fraction_leaf=min_weight_fraction_leaf,
                                           max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                           min_impurity_decrease=min_impurity_decrease,
                                           min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
                                           n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)


class RidgePredictor(Predictor):
    name = 'Ridge Predictor'

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, # noqa
                 tol=0.001, solver='auto', random_state=None, n_jobs=None, name=name):
        super().__init__(name)
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
                           max_iter=max_iter, tol=tol, solver=solver, random_state=random_state)


class LassoPredictor(Predictor):
    name = 'Lasso Predictor'

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=True, copy_X=True, max_iter=1000, tol=0.0001, random_state=None, name=name): # noqa
        super().__init__(name)
        self.model = Lasso(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize,
                           copy_X=copy_X, max_iter=max_iter, tol=tol, random_state=random_state)
