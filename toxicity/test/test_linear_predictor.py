'''Automated tests for the Logistic Predictor class'''
#system settings
import sys
sys.path.append("..") # Append source directory to our Python path

#basics
import numbers
import pandas as pd
import numpy as np
import unittest

#Misc
from linear_predictor import LogisticPredictor
from utils import *
from preprocessing import *



train_file = "../data/train.csv"
test_file = "../data/test.csv"


class TestLinearPredictor(unittest.TestCase):
     #We will use a chunk of the dataset of n rows
     number_of_rows = 1000
     # Logistic predictor parameters
     lr_params = {"C": 4, "dual": True}

     def setUp(self, tf_idf = True):
         self.train =  pd.read_csv(train_file, nrows = TestLinearPredictor.number_of_rows)
         self.test =  pd.read_csv(test_file, nrows = TestLinearPredictor.number_of_rows)
         self.y_train = {tag: self.train[tag].values for tag in TAGS}
         self.logistic_predictor = LogisticPredictor(**TestLinearPredictor.lr_params)
         if tf_idf:
            self.train, self.test = self.idf(self.train, self.test)

     def testStratified(self):
         loss = self.logistic_predictor.evaluate(self.train, self.y_train, method='stratified_CV')
         assert isinstance(loss, numbers.Number)

     def testCV(self):
         loss = self.logistic_predictor.evaluate(self.train, self.y_train, method='CV')
         assert isinstance(loss, numbers.Number)

     def testSplit(self):
         loss = self.logistic_predictor.evaluate(self.train, self.y_train, method='split')
         assert isinstance(loss, numbers.Number)

     def testPredictProba(self):
        '''We will test for the 'toxic' tag'''
        self.logistic_predictor.fit(self.train, self.y_train['toxic'])
        predictions = self.logistic_predictor.predict(self.test)
        assert predictions.shape == (self.test.shape[0], )
        assert (predictions<=1).all() and (predictions>=0).all()


     @staticmethod
     def idf(train_df, test_df):
         return tf_idf(train_df, test_df)

if __name__ == '__main__':
    unittest.main()