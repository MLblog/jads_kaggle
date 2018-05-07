import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import preprocessing

TAGS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def timing(f):
    """
    Decorator to time a function call and print results.
    :param f: Callable to be timed
    :return: Void. Prints to std:out as a side effect
    """

    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        stop = time.time()
        print('{} function took {:.1f} seconds to complete\n'.format(f.__name__, (stop - start)))
        return ret

    return wrap


def scale_data(train, test):
    """
    Creates an scaled version of the train and test sets. This step is necesary to
    use algorithms suc as SVM

    Parameters
    -------------------------
    train and test: test and train sets. Both most have the same structure

    Returns
    --------------------------
    scaled train and test sets
    """
    scaler = preprocessing.StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(test)


def create_submission(predictor, train_x, train_ys, test_x, test_id, write_to):
    """
    Creates a submissions file for the given test set

    :param predictor: The predictor to be used for fitting and predicting
    :param train_x: The (preprocessed) features to be used for fitting
    :param train_ys: A dictionary from tag name to its values in the training set.
    :param test_x: The (preprocessed) features to be used for predicting.
    :param write_to: A file path where the submission is written
    """
    submission = pd.DataFrame({'id': test_id})

    for tag in TAGS:
        print("{} Fitting on {} tag".format(predictor, tag))
        predictor.fit(train_x, train_ys[tag])
        submission[tag] = predictor.predict_proba(test_x)

    submission.to_csv(write_to, index=False)
    print("Submissions created at location " + write_to)


def save_sparse_csr(filename, matrix):
        """
        Save sparce matrices

        Parameters
        -------------------------
        filename: name of the file where the sparce matrix is saved
        matrix: saprce matrix to be saved

        Returns
        --------------------------
        .npz document
        """
        np.savez(filename, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=matrix.shape)


def load_sparse_csr(filename):
        """
        Load sparce matrices

        Parameters
        -------------------------
        filename: name of the file where the sparce matrix is saved

        Returns
        --------------------------
        sparce matrix
        """
        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
