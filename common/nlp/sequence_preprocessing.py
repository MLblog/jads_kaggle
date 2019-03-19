import re
import numpy as np
import pandas as pd
import copy

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# mappings of shorthand phrases to full form
WORD_MAP = {"don't": "do not",
            "won't": "will not",
            "i'm": "i am",
            "he's": "he is",
            "she's": "she is",
            "we're": "we are",
            "they're": "they are",
            "aren't": "are not",
            "doesn't": "does not",
            "didn't": "did not",
            "isn't": "is not",
            "i've": "i have",
            "you've": "you have",
            "they've": "they have",
            "we've": "we have",
            "hasn't": "has not",
            "hadn't": "had not",
            "it's": "it is",
            "that's": "that is",
            "how's": "how is",
            "what's": "what is"}


WORD_MAP_NO_PUNCT = {"dont": "do not",
                     "wont": "will not",
                     "im": "i am",
                     "hes": "he is",
                     "shes": "she is",
                     "were": "we are",
                     "theyre": "they are",
                     "arent": "are not",
                     "doesnt": "does not",
                     "didnt": "did not",
                     "isnt": "is not",
                     "ive": "i have",
                     "youve": "you have",
                     "theyve": "they have",
                     "weve": "we have",
                     "hasnt": "has not",
                     "hadnt": "had not",
                     "thats": "that is",
                     "hows": "how is",
                     "whats": "what is"}

# List of punctuations (obtained from a kaggle kernel)
PUNCTS_LIST = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$',
               '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',
               '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<',
               '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â',
               '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−',
               '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║',
               '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪',
               '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤',
               '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',  '∙', '）', '↓', '、',
               '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔',
               '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]


def preprocess_text_for_dl(*datasets, text_col="question_text",
                           word_map=WORD_MAP, puncts=PUNCTS_LIST,
                           puncts_ignore="", puncts_retain=""):
    """Preprocess strings for use in DL models.

    Performs the following tasks:
    1. lowercase
    2. map shorthand phrases (like "won't") to their full form ("will not")
    3. remove punctuation

    Parameters
    ----------
    datasets: pd.DataFrames to preprocess
        The data in which to clean a text column.
    text_col: str, optional (default: 'question_text')
        The name of the text column in all datasets.
    word_map: dict, optional (default: common.nlp.sequence_preprocessing.WORD_MAP)
        A mapping of words to other words. If not None, the keys are replaced
        with their values. Intended use is to perform step 3 above.
    puncts: list or string (default: common.nlp.sequence_preprocessing.PUNCTS_LIST)
        List of punctions to remove from text
    punct_retain: list or string (default: "")
        List of punctions to retain in text
    puncts_ignore: list or string (default: "")
        List of punctions to ignore (that is, replace by " ")

    Notes
    -----
    - Take into account that the function first lower cases all text and then performs the word mapping
    (before removing punctuation). So, in the mapping, use {"won't": "will not"} instead
    of {"Wont": "Will not}, etc.
    - The method takes an arbitrary number of datasets, so it's possible to preprocess just train or test
    or both. Just make sure all datasets have the same column name for the text column.
    - Preprocessing steps: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings

    Returns
    -------
    new_datasets: tuple of pd.DataFrames
        The datasets with cleaned text columns.
    """
    def clean_string(text):
        """Cleans a single string, i.e., removes certain punctuation,
        retains some and ignores some and makes it lower case."""

        # Remove, retain or ignore(replace by space) punctuations
        text = str(text)
        for punct in puncts_ignore:
            text = text.replace(punct, ' ')
        for punct in puncts_retain:
            text = text.replace(punct, f' {punct} ')
        for punct in puncts:
            text = text.replace(punct, '')

        # Take care of numbers such that they are recognized by embedding
        text = re.sub('[0-9]{5,}', '#####', text)
        text = re.sub('[0-9]{4}', '####', text)
        text = re.sub('[0-9]{3}', '###', text)
        text = re.sub('[0-9]{2}', '##', text)

        # remove multiple spaces, if any
        text = re.sub(' +', ' ', text)

        return text.lower()

    # punctions to remove from string (puncts=puncts-puncts_ignore-puncts_retain)
    puncts = [i for i in puncts if i not in puncts_ignore and i not in puncts_retain]

    # make translation of punctuation characters
    # punctuation_map = str.maketrans('', '', string.punctuation)

    # copy to keep the original data as it is
    new_datasets = copy.deepcopy(datasets)

    for data in new_datasets:
        # lower case
        data[text_col] = data[text_col].str.lower()

        # map shorthand words to their full form using provided mapping
        if word_map:
            regexp = re.compile('|'.join(map(re.escape, word_map.keys())))
            data[text_col] = data[text_col].str.replace(regexp, lambda x: word_map[x.group(0)])

        # remove punctuation
        data[text_col] = data[text_col].apply(lambda x: clean_string(x))

    return new_datasets[0] if len(new_datasets) == 1 else new_datasets


def fit_tokenizer(*datasets, text_col):
    """ Fit the tokenizer based on text in several datasets.

    Parameters
    ----------
    datasets: pd.DataFrames,
        The datasets to tokenize. An arbitrary number of datasets can be processed at once.
    text_col: str
        The name of the column in the datasets that holds the text.

    Returns
    -------
    tokenizer: keras.preprocessing.text.Tokenizer object,
        The fitted Tokenizer instance.
    """
    tokenizer = Tokenizer(filters="", lower=False, split=" ")
    print("Fitting tokenizer on all datasets..")
    tokenizer.fit_on_texts(np.concatenate([data[text_col].values for data in datasets]))

    return tokenizer


def tokenize_and_pad(tokenizer, *datasets, text_col, id_col, max_words=72):
    """ Tokenize text in several datasets and pad to make every vector the same length.

    Parameters
    ----------
    tokenizer: keras.preprocessing.text.Tokenizer object,
        The fitted tokenizer instance to use.
    datasets: pd.DataFrames,
        The datasets to tokenize. An arbitrary number of datasets can be processed at once.
    text_col: str,
        The name of the column in the datasets that holds the text.
    id_col: str,
        The name of the column in the datasets that holds the identifyer.
    max_words: int, optional (default: 72)
        The maximum number of words to use per document.

    Returns
    -------
    tokenized_datasets: tuple of np.arrays,
        The tokenized and padded datasets.
    """
    print("Tokenizing the datasets..")
    results = [pd.DataFrame(np.zeros((len(d), max_words)), index=d[id_col]) for d in datasets]

    for i in range(len(datasets)):
        arr = tokenizer.texts_to_sequences(datasets[i][text_col])
        results[i].loc[:, :] = pad_sequences(arr, maxlen=max_words)

    return results
