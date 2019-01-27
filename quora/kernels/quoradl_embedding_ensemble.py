####################################
### Quora Deep Learning approach ###
####################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from collections import defaultdict
import copy
import re
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNLSTM,
                          CuDNNGRU, Conv1D, Bidirectional, GlobalMaxPool1D)
from keras.models import Sequential, Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
import tensorflow as tf

# to deal with Google embeddings
from gensim.models.keyedvectors import KeyedVectors

# load data with right data types and preprocess
dtypes = {"qid": str, "question_text": str, "target": int}
training = pd.read_csv("../input/train.csv", dtype=dtypes)
testing = pd.read_csv("../input/test.csv", dtype=dtypes)

# load Word2Vec embedding
WORD2VEC_EMBEDDING_FILE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
PARA_EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
GLOVE_EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

# mappings of shorthand phrases to full form
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


def load_embedding_matrix(embedding, word_index):
    """Load a particlar embedding.

    Parameters
    ----------
    embedding: str, one of ['google', 'para', 'glove']
        The embedding matrix to load.
    """
    if embedding == "google":
        embed_dict = load_word2vec(WORD2VEC_EMBEDDING_FILE)
    elif embedding == "para": 
        embed_dict = load_para(PARA_EMBEDDING_FILE)
    elif embedding == "glove": 
        embed_dict = load_glove(GLOVE_EMBEDDING_FILE)
    else:
        raise ValueError("embedding must be one of ['google', 'para', 'glove']. "
                         "Received: {}".format(embedding))

    return create_embedding_matrix(word_index, embed_dict)


def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')


def load_glove(path):
    return defaultdict(lambda: None, (get_coefs(*o.split(" ")) for o in open(path)))


def load_para(path):
    return defaultdict(lambda: None, (get_coefs(*o.split(" ")) for o in open(path, encoding="utf8", errors='ignore') if len(o)>100))


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


def tokenize_and_pad(tokenizer, *datasets, text_col="question_text", id_col="qid", max_words=72):
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
    tokenized_datasets: tuple of np.arrays,
        The tokenized and padded datasets.
    """
    print("Tokenizing the datasets..")
    results = [pd.DataFrame(np.zeros((len(d), max_words)), index=d[id_col]) for d in datasets]
    
    for i in range(len(datasets)):
        arr = tokenizer.texts_to_sequences(datasets[i][text_col])
        results[i].loc[:, :] = pad_sequences(arr, maxlen=max_words)

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
    # print("Boundary: {}. Score: {}%".format(boundary, score))
    return score


def get_best_threshold(val_predictions, val_true, low=0.1, high=0.8, steps=0.01):
    best_b, best_score = 0, 0
    for b in np.arange(low, high, steps):
        score = evaluate_classification_boundary(val_predictions, val_true, b, binary_f1score)
        if score > best_score:
            best_b = b
            best_score = score

    print("Best F1 score is {} at threshold {}".format(best_score, best_b))
    return best_b, best_score


def add_lower(embedding, *raw_datasets):
    vocab = list(fit_tokenizer(*raw_datasets).word_index.keys())
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")
    return embedding


def create_model(embedding_matrix):
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False, input_shape=(max_words,)))
    model.add(Bidirectional(CuDNNLSTM(25, return_sequences=True)))
    # model.add(Bidirectional(CuDNNLSTM(60, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  metrics=["accuracy"],
                  optimizer="adam")

    return model


def seed_everything(seed=1234):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph())
    K.set_session(sess)


# def cross_validate_and_predict(x, y, xtest, folds=5):
#     print("Start cross validation..")
#     kfold = StratifiedKFold(n_splits=folds, random_state=10, shuffle=True)
#     thresholds, scores = [], []
#     probas = np.zeros(len(xtest))
#     for train_index, val_index in kfold.split(x, y):
#         # train, validation split
#         x_train, x_val, y_train, y_val = x[train_index], x[val_index], y[train_index], y[val_index]
#         # create and fit model
#         model = create_model()
#         model.fit(x_train, y_train, batch_size=128, epochs=3,
#                   validation_data=(x_val, y_val), verbose=0)
#         # predict validation labels
#         predicted_y_val = model.predict(x_val).flatten()
#         # get and save best threshold and score
#         boundary, f1score = get_best_threshold(predicted_y_val, y_val)
#         thresholds.append(boundary)
#         scores.append(f1score)
#         print("Fold completed. Validation score = {} at threshold {}".format(f1score, boundary))
#         # predict test set with this model
#         probas += (model.predict(xtest).flatten() / folds)

#     predictions = classify_based_on_probs(probas, boundary=np.mean(thresholds))

#     return predictions


def train_and_predict(train_x, train_y, ensemble_x, test_x, embedding_matrix):
    model = create_model(embedding_matrix)
    model.fit(train_x, train_y, batch_size=128, epochs=1, verbose=0)
    loss, accuracy = model.evaluate(x=train_x, y=train_y, verbose=0)
    print("Training loss: {}, accuracy: {}".format(loss, accuracy))
    # predict validation labels
    yhat_ensemble = model.predict(ensemble_x).flatten()
    yhat_test = model.predict(ensemble_x).flatten()
    del model
    return yhat_ensemble, yhat_test


def train_ensemble(x, y, model_cls=RandomForestClassifier, *y_hats, **model_params):
    """Train ensemble model on several predictions."""
    ensemble = model_cls(**model_params)
    data = pd.concat([x] + [pd.Series(y_hat) for y_hat in y_hats], axis=1)
    print("Shape of ensemble data: {}".format(data.shape))
    ensemble.fit(data, y)
    return ensemble
## END METHODS ##

## Features
seed_everything(seed=1029)
max_words = 72 # maximum number of words in a sentence/document

## Start applying stuff
train_Y = training["target"].copy().values
train, test = preprocess_text_for_dl(training.copy(), testing.copy(), puncts_ignore='/-', puncts_retain='&',
                                     word_map=WORD_MAP)
print("Text preprocessed.")
# tokenize text
tokenizer = fit_tokenizer(train, test)
train_X, test_X = tokenize_and_pad(tokenizer, train, test, text_col="question_text", id_col="qid", max_words=max_words)
print("Text tokenized.")

# train and ensemble split
x_train, x_ensemble, y_train, y_ensemble = train_test_split(train_X, train["target"], shuffle=True, test_size=0.4)
print("Data splitted.")

y_hats_ensemble, y_hats_test = [], []
for embedding in ['glove', 'para', 'google']:
    print("Loading {} embedding..".format(embedding))
    matrix = load_embedding_matrix(embedding, tokenizer.word_index)
    print("Loaded {} embedding. Start training base model.".format(embedding))
    y_hat_ensemble, y_hat_test = train_and_predict(x_train, y_train, x_ensemble, test_X, matrix)
    y_hats_ensemble.append(y_hat_ensemble)
    y_hats_test.append(y_hat_test)
    print("Predictions for {} embedding created.".format(embedding))
    del matrix
    K.clear_session()
    gc.collect()

print("Training ensemble model on hold-out set.")
model_params = {"n_estimators": 1000, "max_depth": 5}
ensemble_model = train_ensemble(x_ensemble, y_ensemble, *y_hats_ensemble, **model_params)

# create final test data
x_test = pd.concat([test_X] + [pd.Series(y) for y in y_hats_test], axis=1)
print("Test data shape: {}".format(x_test.shape))

print("Predicting the test set..")
predictions = ensemble_model.predict(x_test)

submission = pd.DataFrame({"qid": test_X.reset_index()["qid"].values,
                           "prediction": np.array(predictions, dtype=int)})
submission.to_csv("submission.csv", index=False)
print("Submission file save to current working directory.")
