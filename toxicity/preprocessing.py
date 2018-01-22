import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import timing

def check_compatibility(f):
    """
    Decorator to assert that a preprocessing function returns compatible train and test sets.
    Specifically it asserts than the number of columns is equal for the sets.

    :param f: The preprocessing function
    :return: The preprocessed train and test sets
    """
    def wrap(*args):
        train, test = f(*args)
        assert(train.shape[1] == test.shape[1])
        return train, test
    return wrap


@timing
@check_compatibility
def tf_idf(train, test, params=None):
    train["comment_text"].fillna("unknown", inplace=True)
    test["comment_text"].fillna("unknown", inplace=True)

    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    def tokenizer(s): return re_tok.sub(r' \1 ', s).split()

    if not params:
        params = {
            "ngram_range": (1, 2),
            "tokenizer": tokenizer,
            "min_df": 3,
            "max_df": 0.9,
            "strip_accents": 'unicode',
            "use_idf": 1,
            "smooth_idf": 1,
            "sublinear_tf": 1
        }
    vec = TfidfVectorizer(**params)

    train = vec.fit_transform(train["comment_text"])
    test = vec.transform(test["comment_text"])
    return train, test


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train, test = tf_idf(train, test)
    print(train.shape)
    print(test.shape)
