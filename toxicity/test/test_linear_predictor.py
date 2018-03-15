import numbers
import unittest
import pathmagic  # noqa
from linear_predictor import LogisticPredictor
import utils
import pandas as pd
from preprocessing import tf_idf

train_file = "../data/train.csv"
test_file = "../data/test.csv"


class TestLinearPredictor(unittest.TestCase):
    # We will use a chunk of the dataset of n rows
    number_of_rows = 1000
    # Logistic predictor parameters
    lr_params = {"C": 4, "dual": True}

    def setUp(self):
        self.train = pd.read_csv(train_file, nrows=TestLinearPredictor.number_of_rows)
        self.test = pd.read_csv(test_file, nrows=TestLinearPredictor.number_of_rows)
        self.y_train = {tag: self.train[tag].values for tag in utils.TAGS}
        self.logistic_predictor = LogisticPredictor(**TestLinearPredictor.lr_params)
        self.train, self.test, _ = tf_idf(self.train, self.test)

    def test_stratified(self):
        loss = self.logistic_predictor.evaluate(self.train, self.y_train, method='stratified_CV')
        assert isinstance(loss, numbers.Number)

    def test_cv(self):
        loss = self.logistic_predictor.evaluate(self.train, self.y_train, method='CV')
        assert isinstance(loss, numbers.Number)

    def test_split(self):
        loss = self.logistic_predictor.evaluate(self.train, self.y_train, method='split')
        assert isinstance(loss, numbers.Number)

    def test_predict_proba(self):
        """We will test for the 'toxic' tag"""
        self.logistic_predictor.fit(self.train, self.y_train['toxic'])
        predictions = self.logistic_predictor.predict(self.test)
        assert predictions.shape == (self.test.shape[0],)
        assert (predictions <= 1).all() and (predictions >= 0).all()


if __name__ == '__main__':
    unittest.main()
