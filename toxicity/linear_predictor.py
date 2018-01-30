import numpy as np

from sklearn.linear_model import LogisticRegression

from predictor import Predictor


class LogisticPredictor(Predictor):
    """
    Adapted to our class design from the kernel:
    https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb
    """
    name = 'Logistic Regression Predictor'
    def __init__(self, C, dual=True, name=name): # noqa
        super().__init__(name)
        self.model = LogisticRegression(C=C, dual=dual)
        self.C = C
        self.dual = dual
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

        :param test_x: a pd.DataFrame of features to be used for predictions
        :return: The predicted labels
        """
        m = test_x.multiply(self.r)
        return self.model.predict_proba(m)[:, 1]

    def predict(self, test_x):
        m = test_x.multiply(self.r)
        return self.model.predict(m)
