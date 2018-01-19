import numpy as np
from sklearn.linear_model import LogisticRegression

from .predictor import Predictor
from .preprocessing import *


class LogisticPredictor(Predictor):
    """
    Adapted to our class design from the kernel:
    https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb
    """
    def __init__(self, params={}, name='Logistic Regression Predictor'):
        super().__init__(params, name)
        self.set_model(LogisticRegression(**params))
        self.r = {}

    def fit(self, train):

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        train_x = train.drop(self.tags, axis=1)
        for i, tag in enumerate(self.tags):
            train_y = train[tag].values
            self.r[tag] = np.log(pr(1, train_y) / pr(0, train_y))
            self.model[tag].fit(train_x.multiply(self.r[tag]), train_y)

    def predict(self, test_x):
        predictions = np.zeros((len(test_x), len(self.tags)))
        for i, tag in enumerate(self.tags):
            predictions[:, i] = self.model[tag].predict_proba(test_x.multiply(self.r[tag]))[:, 1]

        return predictions

if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Preprocess raw text data
    train, test = tf_idf(train, test)

    lr_params = {"C": 4, "dual": True}
    predictor = LogisticPredictor(**lr_params)

    print("Begin Training " + predictor.name)
    predictor.fit(train)
    predictor.create_submission(test, write_to="submissions/lr_submissions.csv")