import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import timing


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
def tf_idf(train, test, params=None, debug=False):
    """
    Performs Pre_procesing of the data set and tokenization
    Each input is numpy array:
    train: Text to train the model
    test: test to test the model
    params: None by default. It is use to define parameters of the tf_idf model

    Returns:
    train: train set in sparce marix form
    test: test set in sparce matrix form
    vec: tf_idf model
    """
    train["comment_text"].fillna("unknown", inplace=True)
    test["comment_text"].fillna("unknown", inplace=True)

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


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train, test = remove_numbers(train, test)
    train, test = tf_idf(train, test, debug=True)
    print(train.shape)
    print(test.shape)
