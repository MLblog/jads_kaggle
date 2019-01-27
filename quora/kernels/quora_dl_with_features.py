####################################
### Quora Deep Learning approach ###
####################################
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import string
import os
import time
from textblob import TextBlob

from collections import defaultdict
import copy
import re
from sklearn.model_selection import train_test_split

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNLSTM,
                          Concatenate, Conv1D, Bidirectional, GlobalMaxPool1D)
from keras.models import Sequential, Model
from keras import initializers, regularizers, constraints, optimizers, layers

# to deal with Google embeddings
from gensim.models.keyedvectors import KeyedVectors

# load data with right data types and preprocess
dtypes = {"qid": str, "question_text": str, "target": int}
training = pd.read_csv("../input/train.csv", dtype=dtypes)
testing = pd.read_csv("../input/test.csv", dtype=dtypes)

# load Word2Vec embedding
EMBEDDING_FILE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

# Map contractions and common misspellings (e.g., British vs American English)
WORD_MAP = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",
            "could've": "could have", "couldn't": "could not", "didn't": "did not",
            "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is",
            "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
            "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
            "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
            "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
            "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
            "might've": "might have","mightn't": "might not","mightn't've": "might not have",
            "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
            "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
            "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
            "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
            "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
            "she's": "she is", "should've": "should have", "shouldn't": "should not",
            "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is",
            "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
            "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
            "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are",
            "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
            "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
            "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",
            "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
            "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
            "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 
            "y'all'd": "you all would", "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
            "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
            "you're": "you are", "you've": "you have", 'hasnt': 'has not',
            'colour':'color', 'centre':'center', 'didnt':'did not', 'doesnt':'does not',
            'isnt':'is not', 'shouldnt':'should not', 'favourite':'favorite','travelling':'traveling',
            'counselling':'counseling', 'theatre':'theater', 'cancelled':'canceled', 'labour':'labor',
            'organisation':'organization', 'wwii':'world war 2', 'citicise':'criticize', 'instagram': 'social medium',
            'whatsapp': 'social medium', 'snapchat': 'social medium', 'behaviour': 'behavior', 'realise': 'realize',
            'defence': 'defense', 'programme': 'program', 'upvotes': 'votes', 'grey': 'gray', 'btech': 'bachelor of technology',
            'mtech': 'master of technology', 'cryptocurrency': 'digital currency', 'cryptocurrencies': 'digital currencies', 'bitcoin': 'digital currency',
            'bitcoins': 'digital currency', 'Bitcoin': 'digital currency', 'Btech': 'Bachelor of Technology', 'Isnt': 'Is not',
            'Snapchat': 'social medium', 'doesnt': 'does not', 'programmr': 'programmer', 'programr': 'programmer',
            'didnt': 'did not', 'blockchain': 'software technology', 'Shouldnt': 'Should not', 'Doesnt': 'Does not', 'isnt': 'is not',
            'programrs': 'programmers', 'currencys': 'currencies', 'honours': 'honors', 'upvote': 'vote', 'learnt': 'learned', 'licence': 'license',
            'Ethereum': 'digital currency', 'Whatis': 'What is', 'bcom': 'bachelor of commerce', 'aluminium': 'aluminum', 'favour': 'favor',
            'Pinterest': 'social medium', 'cheque': 'check', 'judgement': 'judgment', 'modelling': 'modeling', 'Xiaomi': 'phone', 'Coursera': 'online platform',
            'Quora': 'online platform', 'OnePlus': 'phone', 'wasnt': 'was not', 'recognise': 'recognize', 
            'organisation': 'organization', 'organisations': 'organizations', 'colour': 'color', 'colours': 'colors', 'coloured': 'colored',
            'Fortnite': 'video game', 'centres': 'centers', 
            'Quorans': 'people', "Quoras": "online platform's", 'jewellery': 'jewelry store', 'Lyft': 'ride sharing platform',
            'Didnt': 'Did not', 'practise': 'practice', 'vape': 'smoke', 'WeChat': 'social medium', 'analyse': 'analyze', 'travelled': 'traveled',
            'recognised': 'recognized', 'GDPR': 'privacy bill', 'neighbours': 'neighbors', 'demonetisation': 'demonetization', 'programmes': 'programs',
            'Blockchain': 'software technology', 'Nodejs': 'software technology', 'Coinbase': 'online platform', 'litre': 'liter', 'upvoted': 'voted',
            'sulphuric': 'sulfuric', 'Musks': "Musk's", 'neighbour': 'neighbor', 'selfies': 'photos', 'tyres': 'tires', 'ICOs': 'initial coin offerings',
            'Wasnt': 'Was not', 'realised': 'realized', 'specialisation': 'specialization', 'ethereum': 'digital currency', 'tyre': 'tire',
            'organised': 'organized', 'traveller': 'traveler', 'downvote': 'vote against', 'selfie': 'photo', 'Udacity': 'online platform', 'offence': 'offense',
            'litres': 'liters', 'vapour': 'vapor', 'Qoura': 'online platform', 'fibre': 'fiber', 'aeroplane': 'airplane', 'laymans': 'laymen', 'humour': 'humor',
            'utilise': 'utilize', 'civilisation': 'civilization', 'sulphur': 'sulfur', 'archaeology': 'archeology', 'masterbate': 'masturbate', 'Upwork': 'online platform',
            'neurotypicals': 'non-autistic people', 'criticise': 'criticize', 'organise': 'organize', 'labelled': 'labeled', 'cosx': 'cosine x',
            'judgemental': 'judgmental', 'dreamt': 'dreamed', 'Xamarin': 'medicin', 'MOOCs': 'online classes', 'emojis': 'smileys', 'Unacademy': 'online platform',
            'neighbouring': 'neighboring', 'cancelling': 'canceling', 'numericals': 'numerical', 'honour': 'honor', 'globalisation': 'globalization',
            'practising': 'practicing', 'WooCommerce': 'software technology', 'behavioural': 'behavioral', 'masterbation': 'masturbation', 'AngularJS': 'software technology',
            'wwwyoutubecom': 'online platform'}

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

# Commonly seen bad words
badwords = ['ahole', 'asshole', 'bareback', 'bastard', 'beastial', 'bestial', 'big black', 'bitch', 'black cock', 'chink', 
            'cocks', 'creampie', 'cunt', 'dick', 'feck', 'fondle', 'fuc', 'gays', 'golden shower', 'incest', 'jackass', 'lesbians',
            'lusty', 'moron', 'pedophilia', 'pricks', 'puss', 'raped', 'raping', 'scum', 'shit', 'sissy', 'sluts', 'sodom', 'tits', 
            'tranny', 'transsexual', 'whore']


## utils
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


def remove_stop_words(*datasets, text_col="question_text", stop_words=["to", "a", "of", "and"]):
    """Remove specified words from sentences."""
    for data in datasets:
        data[text_col] = data[text_col].apply(lambda x: " ".join([x for x in x.split() if x not in stop_words]))

    return datasets


def replace_words(*datasets, text_col="question_text", word_map=WORD_MAP):
    """Replace words if they are in the given mapping."""
    def map_word(word):
        try:
            return word_map[word]
        except KeyError:
            return word

    for data in datasets:
        data[text_col] = data[text_col].apply(lambda x: " ".join([map_word(x) for x in x.split()]))

    return datasets


def preprocess_text_for_dl(*datasets, text_col="question_text",
                           word_map=WORD_MAP, puncts=PUNCTS_LIST,
                           puncts_ignore="/-", puncts_retain="&",
                           stop_words=["to", "a", "of", "and", "To", "A", "Of"],
                           replace_before=True, replace_after=True):
    """Preprocess strings for use in DL models.

    Performs the following tasks:
    1. ...

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

        return text

    # punctions to remove from string (puncts=puncts-puncts_ignore-puncts_retain)
    puncts = [i for i in puncts if i not in puncts_ignore and i not in puncts_retain]

    # copy to keep the original data as it is
    new_datasets = copy.deepcopy(datasets)

    # replace misspellings and contractions
    if replace_before:
        new_datasets = replace_words(*new_datasets, text_col=text_col, word_map=word_map)

    # remove punctuation
    for data in new_datasets:
        data[text_col] = data[text_col].apply(lambda x: clean_string(x))

    # replace misspellings and contractions again, now that punctuation is removed
    if replace_after:
        new_datasets = replace_words(*new_datasets, text_col=text_col, word_map=word_map)

    # remove stop words that are not in the embedding
    if stop_words:
        new_datasets = remove_stop_words(*new_datasets, text_col=text_col, stop_words=stop_words)

    return new_datasets[0] if len(new_datasets) == 1 else new_datasets


## Classes
class FeatureAdder(object):
    def __init__(self, data_dir="data", upper_case=False, word_count=False, unique_words_count=False,
                 letter_count=False, punctuation_count=False, little_case=False, stopwords=False,
                 question_or_exclamation=False, number_bad_words=False, sentiment_analysis=False,
                 text_column="comment_text", badwords=None):

        self.text_column = text_column
        self.badwords = badwords
        self.data_dir = data_dir
        self.features = {
            self._upper: upper_case,
            self._count_words: word_count,
            self._unique_words: unique_words_count,
            self._count_letters: letter_count,
            self._count_punctuation: punctuation_count,
            self._count_little_case: little_case,
            self._count_stopwords: stopwords,
            self._question_or_exclamation: question_or_exclamation,
            self._count_bad_words: number_bad_words,
            self._polarity_subjectivity_score: sentiment_analysis
            }

    def _polarity_subjectivity_score(self, df):
        """
        This method calculates the polarity and subjetivity score. Both metrics are used to
        evaluate the sentiment of a text.
        Parameters
        -------------------------
        df: Pandas Dataframe. Assumed to contain text in a column named `self.text_column`
        Returns
        --------------------------
        df: Data frame with the polarity and sbjectivity score
        """
        def polarity_score(x):
            return TextBlob(x).sentiment.polarity

        def subjectivity_score(x):
            return TextBlob(x).sentiment.subjectivity

        df['polarity_score'] = df[self.text_column].apply(lambda x: polarity_score(x))
        df['subjectivity_score'] = df[self.text_column].apply(lambda x: subjectivity_score(x))
        return df

    def set_path(self, data_dir):
        self.data_dir = data_dir

    def _count_bad_words(self, df):
        """
        This is a method that creates a new feature with the number of words in the google bad list.
        Source: https://www.freewebheaders.com/full-list-of-bad-words-banned-by-google/
        Parameters
        -------------------------
        df: Pandas Dataframe. Assumed to contain text in a column named `self.text_column`
        Returns
        --------------------------
        df: Data frame with the number of the bad words according to the google dictionary.
        """

        def union_sets(*dfs):
            final = set()
            for df in dfs:
                s = set(df[list(df)[0]])  # Assuming that curse words are the 1st column
                final = final.union(s)
            return [' {} '.format(x) for x in final]

        def count_badwords(comment):
            try:
                return sum(badword.lower() in comment.lower() for badword in badwords)
            except AttributeError:
                # Bad comment, probably NaN
                return 0

        # If the list of badwords is not given, then external data is used.
        if self.badwords is None:
            badwords_1_path = os.path.join(self.data_dir, "badwords", "google_bad_words.csv")
            badwords_2_path = os.path.join(self.data_dir, "badwords", "bad_words.csv")

            try:
                badwords_1 = pd.read_csv(badwords_1_path, 'utf-8', engine="python")
                badwords_2 = pd.read_csv(badwords_2_path, sep=',')
            except FileNotFoundError:
                print("Could not find the badwords folder at {}\n"
                      "Please provide the data root path using the `set_path` method.".format(self.data_dir))
                return None

            badwords = union_sets(badwords_1, badwords_2)
        else:
            badwords = self.badwords

        df["count_bad_words"] = df[self.text_column].apply(count_badwords)
        return df

    def _upper(self, df):
        """
        This is a method that creates a new feature with the number of capitalized words.

        Parameters
        -------------------------
        df: pd.Dataframe. Assumed to contain text in a column named `self.text_column`

        Returns
        --------------------------
        pd.Dataframe with the number of the capitalized words as an extra feature.
        """
        df["count_words_upper"] = df[self.text_column].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        return df

    def _count_words(self, df):
        """
        This is a method that creates a new feature with the number of words in the sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `self.text_column`

        Returns
        --------------------------
        pd.Dataframe with the number of words as an extra feature.
        """
        df['count_word'] = df[self.text_column].apply(lambda x: len(str(x).split()))
        return df

    def _unique_words(self, df):
        """
        This is a method that creates a new feature with the number of unique words in the sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `self.text_column`

        Returns
        --------------------------
        pd.Dataframe with the number of the unique words as an extra feature.
        """
        df['count_unique_word'] = df[self.text_column].apply(lambda x: len(set(str(x).split())))
        return df

    def _count_letters(self, df):
        """
        This is a method that creates a new feature with the number of letters in the sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `self.text_column`

        Returns
        --------------------------
        pd.Dataframe with the aggregated number of the characters as an extra feature.
        """
        df['count_letters'] = df[self.text_column].apply(lambda x: len(str(x)))
        return df

    def _count_punctuation(self, df):
        """
        This is a method that creates a new feature with the number of the punctuation marks.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `self.text_column`

        Returns
        --------------------------
        pd.Dataframe with the number of the puncutation symbols as an extra feature.
        """
        df["count_punctuations"] = df[self.text_column].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation]))
        return df

    def _count_little_case(self, df):
        """
        This is a method that creates a new feature with the number of the lowercase words.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `self.text_column`

        Returns
        --------------------------
        pd.Dataframe with the number of the not capitalized words as an extra feature.
        """
        df["count_words_title"] = df[self.text_column].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
        return df

    def _count_stopwords(self, df):
        """
        This is a method that creates a new feature with the number of the stop words in a sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `self.text_column`

        Returns
        --------------------------
        pd.Dataframe with the number of the stop words (like 'then', 'to', 'a' etc) as an extra feature.
        """
        df["count_stopwords"] = df[self.text_column].apply(
            lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
        return df

    def _question_or_exclamation(self, df):
        """
        This is a method that creates a new feature with the number of the question or exclamation marks used in the sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `self.text_column`

        Returns
        --------------------------
        pd.Dataframe with the number of the question or exclamation marks as an extra feature.
        """
        df['question_mark'] = df[self.text_column].str.count('\\?')
        df['exclamation_mark'] = df[self.text_column].str.count('\\!')
        return df

    @timing
    def get_features(self, train=None, test=None, save=False, load=True):
        """
        Call feature extractors that have been activated (by setting their boolean attribute to True)

        Parameters
        -------------------------
        train, test: pd.Dataframes to go through the transformation
        save: boolean. True to save the train and test data set in your local machine
        load: boolean. True to get the train and test set from your local machine

        Returns
        --------------------------
        (pd.DataFrame, pd.DataFrame)
                DataFrames containing all features

        Example
        -------
            >>> # Activate desired feature extraction
            >>> params = {'upper_case':True, 'word_count':True, 'unique_words_count':True,
                 'letter_count':True, 'punctuation_count':True, 'little_case':True,
                 'stopwords':True, 'question_or_exclamation':True, 'number_bad_words':True}
            >>> fa = FeatureAdder(**params)
            >>>
            >>> # to create and save train and test in your local machine
            >>> train, test = fa.get_features(df_train, df_test, load=False, save=True)
            >>>
            >>> # to load results from your local machine
            >>> train, test = fa.get_features(load=True)

        """
        base_dir = self.data_dir + "/output"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        name_train = base_dir + '/df_train_features_added.csv'
        name_test = base_dir + '/df_test_features_added.csv'

        if load:
            if os.path.exists(name_train) and os.path.exists(name_test):
                print('getting files from your local machine')
                train, test = pd.read_csv(name_train), pd.read_csv(name_test)
            else:
                raise ValueError("You asked to load the features but they were not found " +
                                 "at the specified location: \n{}\n{}".format(name_train, name_test))
        else:
            print('Computing the new features, this will take a while...!')
            for method, condition in self.features.items():
                if condition:
                    method(train), method(test)

            train.drop(self.text_column, axis=1, inplace=True)
            test.drop(self.text_column, axis=1, inplace=True)

        if save:
            print('Saving train file as {}'.format(name_train))
            train.to_csv(name_train, index=False)
            print('Saving test file as {}'.format(name_test))
            test.to_csv(name_test, index=False)
            print('Files saved')

        return train, test


def load_word2vec(path, return_as_dict=True):
    """ Load an return Word2Vec embeddings.

    Parameters
    ----------
    path: str,
        The path to the Word2Vec embedding file.
    return_as_dict: boolean, optional (default: True),
        Whether to return a normal dictionary (True) or a gensim KeyedVectors object (False).

    Returns
    -------
    Dict or KeyedVectors object containing the word embeddings.
    """
    word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
    word_vectors = word2vec.wv

    if return_as_dict:
        # copy to normal dictionary
        embedding_dict = defaultdict(lambda: None)
        for word in word_vectors.vocab:
            embedding_dict[word] = word_vectors[word]
        return embedding_dict
    else:
        return word_vectors


def fit_tokenizer(*datasets, text_col="question_text"):
    """ Tokenize text in several datasets and pad to make every vector the same length.

    Parameters
    ----------
    datasets: pd.DataFrames,
        The datasets to tokenize. An arbitrary number of datasets can be processed at once.
    text_col: str, optional (default: 'question_text')
        The name of the column in the datasets that holds the text.

    Returns
    -------
    tokenizer: keras.preprocessing.text.Tokenizer object,
        The fitted Tokenizer instance.
    """
    tokenizer = Tokenizer()
    print("Fitting tokenizer on all datasets..")
    tokenizer.fit_on_texts(np.concatenate([data[text_col].values for data in datasets]))
    return tokenizer


def tokenize_and_pad(tokenizer, *datasets, text_col="question_text", id_col="qid", max_words=60):
    """ Tokenize text in several datasets and pad to make every vector the same length.

    Parameters
    ----------
    tokenizer: keras.preprocessing.text.Tokenizer object,
        The fitted tokenizer instance to use.
    datasets: pd.DataFrames,
        The datasets to tokenize. An arbitrary number of datasets can be processed at once.
    text_col: str, optional (default: 'question_text')
        The name of the column in the datasets that holds the text.
    max_words: int, optional (default: 60)
        The maximum number of words to use per document.

    Returns
    -------
    tokenized_datasets: tuple of pd.DataFrames,
        The tokenized and padded datasets.
    """
    print("Tokenizing the datasets..")
    results = [pd.DataFrame(np.empty((len(d), max_words)), index=d[id_col]) for d in datasets]
    
    for i in range(len(datasets)):
        results[i].loc[:, :] = pad_sequences(tokenizer.texts_to_sequences(datasets[i][text_col]), maxlen=max_words)

    return results


def create_embedding_matrix(word_index, embedding):
    """ Create a matrix of weights to use in an embedding layer.

    Parameters
    ----------
    tokenizer: dict like {word -> index (unique int)},
        The tokenizer that was fitted on the relevant data.
    embedding: dict like {word -> vector},
        The pretrained embedding to use.

    Returns
    -------
    embedding_matrix: np.ndarray,
        The matrix of word vectors to use as weights in an Embedding layer.
    """
    mu, sigma = np.mean(list(embedding.values())), np.std(list(embedding.values()))
    embed_size = embedding[list(embedding.keys())[0]].shape[0]
    print("Embedding size is {}.".format(embed_size))
    matrix = np.random.normal(mu, sigma, (len(word_index) + 1, embed_size))

    counter = 0
    for word, i in word_index.items():
        vec = embedding[word]
        if vec is not None:
            matrix[i] = vec
            counter += 1

    print("Found {} embeddings of {} total words in vocabulary.".format(counter, len(word_index)))
    return matrix


def binary_f1score(y_predict, y_true):
    """F1 score for boolean / binary problems"""
    true_positives = np.sum((y_predict == 1) & (y_true == 1))
    false_positives = np.sum((y_predict == 1) & (y_true == 0))
    true_negatives = np.sum((y_predict == 0) & (y_true == 0))
    false_negatives = np.sum((y_predict == 0) & (y_true == 1))
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f = 2 * (precision * recall) / (precision + recall)
    return f


def classify_based_on_probs(predictions, boundary=0.2):
    labels = (predictions > boundary)
    return np.array(labels, dtype=int)


def evaluate_classification_boundary(predictions, true_labels, boundary, evaluation_func):
    predicted_labels = classify_based_on_probs(predictions, boundary=boundary)
    score = evaluation_func(predicted_labels, true_labels)
    print("Boundary: {}. Score: {}%".format(boundary, score))
    return score


def normalize_features(*datasets):
    # scale features between -1 and 1 over train and test together
    maxes = pd.concat([data for data in datasets], sort=True).max(axis=0)
    n = len(maxes)
    new_datasets = copy.deepcopy(datasets)
    for data in new_datasets:
        data.loc[:, :] = 2 * (data.values / maxes.values.reshape((1, n))) - 1 # make between -1 and 1

    return new_datasets


def create_two_legged_model():
    ## Create and fit two-legged Neural Network
    input1 = Input(shape=(max_words,))
    rnn = Embedding(len(tokenizer.word_index) + 1, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False, input_shape=(max_words,))(input1)
    rnn = Bidirectional(CuDNNLSTM(60, return_sequences=True))(rnn)
    rnn = GlobalMaxPool1D()(rnn)
    rnn = Dropout(0.1)(rnn)
    
    input2 = Input(shape=(train_x2.shape[1],))
    nn = Dense(64, activation="relu")(input2)
    nn = Dense(64, activation="relu")(nn)
    nn = Dense(64, activation="relu")(nn)
    nn = Dropout(0.2)(nn)
    
    merged = Concatenate(axis=-1)([rnn, nn])
    merged = Dense(64, activation="relu")(merged)
    merged = Dense(64, activation="relu")(merged)
    merged = Dropout(0.1)(merged)
    output = Dense(1, activation="sigmoid")(merged)
    
    model = Model(inputs=[input1, input2], outputs=output)
    
    model.compile(loss="binary_crossentropy",
                  metrics=["accuracy"],
                  optimizer="adam")
    print("Model created and compiled.")
    return model
## END METHODS ##

print("Start running...")
## Features
max_words = 72 # maximum number of words in a sentence/document
fa_params = {
    "data_dir": 'Data/',
    "upper_case": True,
    "word_count": True,
    "unique_words_count": True,
    "letter_count": True,
    "punctuation_count": True,
    "little_case": True,
    "stopwords": False,
    "question_or_exclamation": True,
    "number_bad_words": True,
    "sentiment_analysis": True,
    "badwords": badwords,
    "text_column": "question_text"
}
fa = FeatureAdder(**fa_params)

## Start applying stuff
# get features
train_extended, test_extended = fa.get_features(training.copy(), testing.copy(), load=False, save=False)

train_extended.set_index("qid", inplace=True)
test_ids = test_extended["qid"].copy().values
test_extended.set_index("qid", inplace=True)

train_target = train_extended["target"].copy().values
train_extended.drop("target", inplace=True, axis=1)

train_features, test_features = normalize_features(train_extended, test_extended)

del train_extended, test_extended
gc.collect()
print("Features obtained and normalized. Garbage collected")

# preprocess for DL
train, test = preprocess_text_for_dl(training, testing, puncts_ignore='/-', puncts_retain='&',
                                     word_map=WORD_MAP)
tokenizer = fit_tokenizer(train, test)

del training, testing
gc.collect()
print("Tokenizer fitted. Garbage collected")

train_sequence, test_sequence = tokenize_and_pad(tokenizer, train, test, text_col="question_text", id_col="qid", max_words=max_words)

del train, test
gc.collect()
print("Texts sequences obtained and padded. Garbage collected")

# train, validation, test datasets
train_x1, val_x1, train_x2, val_x2, train_y, val_y = train_test_split(train_sequence,
                                                                      train_features,
                                                                      train_target,
                                                                      shuffle=True,
                                                                      test_size=0.1,
                                                                      stratify=train_target)

test_x1 = test_sequence
test_x2 = test_features

del train_sequence, train_features, train_target
gc.collect()
print("Train, validation splits made. Garbage collected.")

# load embeddings and create matrix
embedding_dict = load_word2vec(EMBEDDING_FILE, return_as_dict=True)
embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict)

del embedding_dict
gc.collect()
print("Embedding matrix created. Garbage collected.")

model = create_two_legged_model()

print("Model compiled. Start training.")
hist = model.fit([train_x1, train_x2], train_y, batch_size=64, epochs=5,
                 validation_data=([val_x1, val_x2], val_y), verbose=False)

## Create submission
# Choose best classification boundary
val_predictions = model.predict([val_x1, val_x2])
val_predictions = val_predictions.flatten()

best_b, best_score = 0, 0
for b in np.arange(0.1, 0.9, 0.01):
    score = evaluate_classification_boundary(val_predictions, val_y, b, binary_f1score)
    if score > best_score:
        best_b = b
        best_score = score

print("Best F1 score is {} with boundary of {}.".format(best_score, best_b))

# Predict the test set and submit
predictions = model.predict([test_x1, test_x2])
predicted_labels_test = classify_based_on_probs(predictions, boundary=best_b)
submission = pd.DataFrame({"qid": test_ids,
                           "prediction": np.array(predicted_labels_test.flatten(), dtype=int)})
submission.to_csv("submission.csv", index=False)
print("Submission file saved to current working directory.")
