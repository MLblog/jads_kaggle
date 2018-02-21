import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import timing, save_sparse_csr, load_sparse_csr
import os

def check_compatibility(f):
    """
    Decorator to assert that a preprocessing function returns compatible train and test sets.
    Specifically it asserts than the number of columns is equal for both sets.

    :param f: The preprocessing function
    :return: The preprocessed train and test sets
    """
    def wrap(*args, **kwargs):
        train, test = f(*args, **kwargs)
        assert(train.shape[1] == test.shape[1])
        return train, test
    return wrap


def remove_numbers(train, test):
    def remove_numbers_helper(s):
        return s.apply(lambda s: ''.join([i for i in s if not i.isdigit()]))

    train["comment_text"] = remove_numbers_helper(train["comment_text"])
    test["comment_text"] = remove_numbers_helper(test["comment_text"])
    return train, test


@check_compatibility
@timing
def tf_idf(train, test, params=None, remove_numbers_function=True, debug=False):
    """
    Performs Pre_procesing of the data set and tokenization
    Each input is numpy array:
    train: Text to train the model
    test: test to test the model
    params: None by default. It is use to define parameters of the tf_idf model
    remove_numbers_function: True if removing numbers is desired

    Returns:
    train: train set in sparce marix form
    test: test set in sparce matrix form
    """
    train["comment_text"].fillna("unknown", inplace=True)
    test["comment_text"].fillna("unknown", inplace=True)

    if remove_numbers_function:
        train, test = remove_numbers(train, test)

    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    def tokenizer(s): return re_tok.sub(r' \1 ', s).split()

    if not params:
        params = {
            "ngram_range": (1, 2),
            "tokenizer": tokenizer,
            "min_df": 0.0001,
            "max_df": 0.995,
            "strip_accents": 'unicode',
            "use_idf": 1,
            "smooth_idf": 1,
            "sublinear_tf": 1
        }
    vec = TfidfVectorizer(**params)

    train = vec.fit_transform(train["comment_text"])
    test = vec.transform(test["comment_text"])

    if debug:
        print("Removing these tokens:\n{}".format(vec.stop_words_))

    return train, test

def get_sparse_matrix(train=None, test=None, params=None, remove_numbers_function=True, debug=True, save=False, load=True, data_dir="data"):    
    """
    Get sparse matrix form of the train and test set

    Parameters
    -------------------------
    Each input is numpy array:
    train, test, params: See the documentation of tf_idf function
    save: To save the train and test sprse matrices on . npz format
    load: To load the train and test sprse matrices from your local machine
    data_dir: Specify the ro specifi the data directory where the matrices are saved

    Returns:
    --------------------------
    train: train set in sparce marix form
    test: test set in sparce matrix form

    Example
    -------
        >>> train = pd.read_csv("data/train.csv")
        >>> test = pd.read_csv("data/test.csv")
        >>> # to create and save the train and test set
        >>> train_sparse, test_sparse = get_sparse_matrix(train, test, params=None, remove_numbers_function=True, debug=True, save=True, load=False)
        >>> # to load the sparse matrices from your local machine
        >>> train, test = get_sparse_matrix(load=True)
    """
    base_dir = data_dir + '/output/'

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    name_train = base_dir + 'sparce_train.npz'
    name_test = base_dir + 'sparce_test.npz'

    if load:
        if os.path.exists(name_train) and os.path.exists(name_test):
            train, test = load_sparse_csr(name_train), load_sparse_csr(name_test)
        else:
            raise ValueError("You asked to load the features but they were not found " +
                                 "at the specified location: \n{}\n{}".format(name_train, name_test))

    else:
        print('Computing the sparse matrixes, this will take a while...!')
        train, test = tf_idf(train, test, params, remove_numbers_function, debug)

    if save:
        print('Saving train file as {}'.format(name_train))
        save_sparse_csr(name_train, train)
        print('Saving test file as {}'.format(name_test))
        save_sparse_csr(name_test, test)

    return train, test

if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    get_sparse_matrix(train, test, params=None, remove_numbers_function=True, debug=True, save=True, load=False)
    #train, test = get_sparse_matrix(load=True)

