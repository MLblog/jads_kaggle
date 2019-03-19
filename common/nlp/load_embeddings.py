import numpy as np
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors


def load_word_embedding(filepath):
    """Given a filepath to embeddings file, return a word to vec dictionary.
    E.g. {'word': array([0.1, 0.2, ...])}
    Parameters
    ----------
    filepath: str
        The path to the embeddings file

    Notes
    -----
    -   On the following websites embeddings are found that can be used as input:
            https://nlp.stanford.edu/projects/glove/
            https://cogcomp.org/page/resource_view/106

    Returns
    -------
    The word to vec dictionary
    """
    def _get_vec(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    print('load word embedding ......')
    try:
        word_embedding = dict(_get_vec(*w.split(' ')) for w in open(filepath))
    except UnicodeDecodeError:
        word_embedding = dict(_get_vec(*w.split(' ')) for w in open(
            filepath, encoding="utf8", errors='ignore'))

    # Sanity check for word vector length
    words_to_del = []
    for word, vec in word_embedding.items():
        if len(vec) != 300:
            words_to_del.append(word)
    for word in words_to_del:
        del word_embedding[word]

    return word_embedding


def load_word2vec(filepath, return_as_dict=True):
    """ Given a filepath to embeddings file, return a word to vec dictionary or KeyedVector.
    E.g. {'word': array([0.1, 0.2, ...])}

    Parameters
    ----------
    filepath: str,
        The path to the embeddings file.
    return_as_dict: boolean, optional (default: True),
        Whether to return a normal dictionary (True) or a gensim KeyedVectors object (False).

    Notes
    -----
    -   On the following websites embeddings are found that can be used as input:
            https://code.google.com/archive/p/word2vec/

    Returns
    -------
    Dict or KeyedVectors object containing the word embeddings.
    """
    word2vec = KeyedVectors.load_word2vec_format(filepath, binary=True)
    word_vectors = word2vec.wv

    if return_as_dict:
        embedding_dict = defaultdict(lambda: None)
        for word in word_vectors.vocab:
            embedding_dict[word] = word_vectors[word]
        return embedding_dict
    else:
        return word_vectors


def create_embedding_matrix(word_index, embedding):
    """ Create a matrix of weights to use in an embedding layer.

    Parameters
    ----------
    word_index: dict like {word -> index (unique int)},
        The tokenizer that was fitted on the relevant data.
    embedding: dict like {word -> vector},
        The pretrained embedding to use.

    Returns
    -------
    embedding_matrix: np.ndarray,
        The matrix of word vectors to use as weights in an Embedding layer.
    """

    # Initialize
    mu, sigma = np.mean(list(embedding.values())), np.std(list(embedding.values()))
    embed_size = embedding[list(embedding.keys())[0]].shape[0]
    matrix = np.random.normal(mu, sigma, (len(word_index) + 1, embed_size))

    counter = 0
    for word, i in word_index.items():
        vec = embedding.get(word)
        if vec is not None:
            matrix[i] = vec
            counter += 1

    print("Found {} embeddings of {} total words in vocabulary.".format(counter, len(word_index)))
    return matrix
