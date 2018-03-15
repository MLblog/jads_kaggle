import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim import corpora, models

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
    """Removes numbers - who would have guessed!"""
    def remove_numbers_helper(s):
        return s.apply(lambda s: ''.join([i for i in s if not i.isdigit()]))

    train["comment_text"] = remove_numbers_helper(train["comment_text"])
    test["comment_text"] = remove_numbers_helper(test["comment_text"])
    return train, test


@check_compatibility
@timing
def gensim_preprocess(train, test, model_type='lsi', num_topics=500,
                      use_own_tfidf=False, force_compute=False, report_progress=False,
                      data_dir='data/', **tfidf_params):

    """Use topic modeling to create a dense matrix representation of the input text.

    Notes
    -----
    In many cases I create artifacts (corpora, tfidf representations etc.) for:
        1. Training set
        2. Test set
        3. Their concatenation

    The concatenation is used to create the models, i.e compute embeddings since the labels are not needed in this unsupervised stage.
    The first two are used for the training and evaluation/submission stages accordingly.

    Parameters
    ----------
    :param train: The training set as a pd.Dataframe including the free text column "comment_text".
    :param test: The test set as a pd.Dataframe including the free text column "comment_text".
    :param model: Dimensionality reduction model to be used, can be 'lsi', 'lda' for now (more might be added later).
    :param num_topics: Number of columns (features) in the output matrices.
    :use_own_tfidf: If true, our own version of tfidf will be used with **tfidf_params passed to it
    :force_compute: If True we will not even try to load but instead compute everything. Set it if you want to try
                    different parameters.
    :report_progress: If True, progress will be reported when each computationally expensive step is starting.
    :data_dir: Path to the base data directory. Used to call this method from anywhere.
               For example a notebook would provide `data_dir='../data'`
    :**tfidf_params: Key-Value parameters passed to our own `tf_idf` implementation.
                     Only used if `use_own_tfidf` is set to True.

    Returns
    -------
    :return: (train, test) datasets as 2D np.ndarrays of shape (num_comments, `num_topics`)
    """

    # Folder where gensim models and data will be saved to and loaded from.
    gensim_dir = data_dir + 'gensim/'

    def progress(msg):
        """Helper to conditionally print progress messages to std:out."""
        if report_progress:
            print(msg)

    if force_compute:
        progress("This is gonna take a while mate, grab a coffee/beer. Actually you might wanna take a walk as well. Or a nap :D")

    train_text = train["comment_text"].tolist()
    test_text = test["comment_text"].tolist()

    # Tokenize
    def safe_tokenize(comment):
        """Wrap `nltk.word_tokenize` but also handle corrupted input."""
        try:
            return nltk.word_tokenize(comment)
        except TypeError:
            return ["UNKNOWN"]

    progress("Tokenizing text, this will take a while...")
    train_texts = [safe_tokenize(comment) for comment in train_text]
    test_texts = [safe_tokenize(comment) for comment in test_text]

    dictionary = corpora.Dictionary(train_texts + test_texts)

    # Lets create the TF-IDF representation needed for the dimensionality reduction models.
    if use_own_tfidf:
        # Untested yet but I hope it works. I mean, why wouldn't it right?
        progress("Using our own version of TF-IDF, this will take a while...")
        train_tfidf, test_tfidf, whole_tfidf = tf_idf(train, test, **tfidf_params)

    else:
        # Use gensims TFIDF model - Tested while under the influence of 10 beers.
        # I code well when drunk though so no worries.

        # Read or create the corpus
        try:
            # Hack to redirect to the exception handler - yes I know its bad but I like it mmmkay?
            if force_compute:
                raise FileNotFoundError
            train_corpus = corpora.MmCorpus(gensim_dir + 'training_corpus.mm')
            test_corpus = corpora.MmCorpus(gensim_dir + 'test_corpus.mm')
            whole_corpus = corpora.MmCorpus(gensim_dir + 'whole_corpus')
        except FileNotFoundError:
            progress("Creating the gensim corpora, this will take a while...")
            train_corpus = [dictionary.doc2bow(comment) for comment in train_texts]
            test_corpus = [dictionary.doc2bow(comment) for comment in test_texts]
            whole_corpus = [dictionary.doc2bow(comment) for comment in train_texts + test_texts]
            corpora.MmCorpus.serialize(gensim_dir + 'training_corpus.mm', train_corpus)
            corpora.MmCorpus.serialize(gensim_dir + 'test_corpus.mm', test_corpus)
            corpora.MmCorpus.serialize(gensim_dir + 'whole_corpus.mm', whole_corpus)

        progress("Using gensim's implementation of TF-IDF, this will take a while...")
        tfidf_model = models.TfidfModel(whole_corpus)
        train_tfidf = tfidf_model[train_corpus]
        test_tfidf = tfidf_model[test_corpus]
        whole_tfidf = tfidf_model[train_corpus + test_corpus]

    # Feed the TF-IDF representation to the dimensionality reduction model - this is slow so try to load it first.
    if model_type == 'lsi':
        try:
            # Hack to redirect to the exception handler - yes I know its bad but I like it mmmkay?
            if force_compute:
                raise FileNotFoundError
            model = models.LsiModel.load(gensim_dir + 'lsi.model')
        except FileNotFoundError:
            progress("Creating the LSI model, this will take a while...")
            model = models.LsiModel(whole_tfidf, id2word=dictionary, num_topics=num_topics)
            model.save(gensim_dir + 'lsi.model')

    elif model_type == 'lda':
        try:
            # Hack to redirect to the exception handler - yes I know its bad but I like it mmmkay?
            if force_compute:
                raise FileNotFoundError
            model = models.LdaModel.load('data/lda.model')
        except FileNotFoundError:
            progress("Creating the LDA model, this will take a while...")
            model = models.LdaModel(whole_tfidf, id2word=dictionary, num_topics=num_topics)
            model.save(gensim_dir + 'lda.model')

    else:
        raise ValueError("Only 'lda' and 'lsi' models are supported, you passed {}".format(model_type))

    train = model[train_tfidf]
    test = model[test_tfidf]

    # Transform into a 2D array format.
    print("Reformatting output to a 2D array, this will take a while...")
    values = np.vectorize(lambda x: x[1])
    return values(np.array(train)), values(np.array(test))


@check_compatibility
@timing
def truncatedsvd_preprocess(train, test, num_topics=500, report_progress=False,
                            use_own_tfidf=True, data_dir='data/', save=False, **tfidf_params):

    """ Use Latent Semantic Analysis (LSA/LSI) to obtain a dense matrix representation of the input text.

    Parameters
    ----------
    :param train: The training set as a pd.Dataframe including the free text column "comment_text".
    :param test: The test set as a pd.Dataframe including the free text column "comment_text".
    :param num_topics: Number of columns (features) in the output matrices.
    :report_progress: If True, progress will be reported when each computationally expensive step is starting.
    :use_own_tfidf: If True, uses our own implementation of tfidf.
    :data_dir: Path to the base data directory. Used to call this method from anywhere.
               For example a notebook would provide `data_dir='../data'`

    Returns
    -------
    :return: (train, test) datasets as 2D np.ndarrays of shape (num_comments, `num_topics`)
    """

    def progress(msg):
        """Helper to conditionally print progress messages to std:out."""
        if report_progress:
            print(msg)

    # create lists of comments/strings
    train_text = train["comment_text"].tolist()
    test_text = test["comment_text"].tolist()
    all_text = train_text + test_text

    # create the TF-IDF representation needed for dimensionality reduction.
    if use_own_tfidf:
        # Untested yet but I hope it works. I mean, why wouldn't it right?
        progress("Using our own version of TF-IDF, this will take a while...")
        train_tfidf, test_tfidf, whole_tfidf = tf_idf(train, test, **tfidf_params)

    else:
        # use sklearn's TF-IDF in combination with NLTK's tokenizer
        progress("Creating TF-IDF model and representations..")
        tfidf_model = TfidfVectorizer(input='content',
                                      encoding='utf-8',
                                      decode_error='strict',
                                      lowercase=True,
                                      tokenizer=nltk.word_tokenize,
                                      analyzer='word',
                                      stop_words=None)

        tfidf_model.fit(all_text)
        train_tfidf = tfidf_model.transform(train_text)
        test_tfidf = tfidf_model.transform(test_text)
        whole_tfidf = tfidf_model.transform(all_text)

    # Feed the TF-IDF representation to the dimensionality reduction model.
    progress("Fitting SVD to all data..")
    svd = TruncatedSVD(n_components=num_topics, n_iter=7)
    svd.fit(whole_tfidf)

    progress("Transforming train and test sets..")
    x_train = svd.transform(train_tfidf)
    x_test = svd.transform(test_tfidf)

    # save and return data
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    if save:
        x_train.to_csv(data_dir+"train_"+str(int(num_topics))+".csv")
        x_test.to_csv(data_dir+"test_"+str(int(num_topics))+".csv")

    progress("Dimensionality reduction completed.")
    return x_train, x_test


@timing
def tf_idf(train, test, params=None, remove_numbers_function=True, debug=False, stemming=True, lemmatization=False):
    """
    Performs preprocessing of the data set and tokenization
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

    if lemmatization + stemming == 2:
        raise ValueError("It is not possible to apply both stemming and lemmatization. Please choose one of them.")
    if stemming:
        def tokenizer(s):
            stemmer = nltk.stem.PorterStemmer()
            tokens = nltk.word_tokenize(s)
            stems = []
            for item in tokens:
                try:
                    stems.append(stemmer.stem(item))
                except RecursionError:
                    stems.append('Big_word')
            return stems
    elif lemmatization:
        def tokenizer(s):
            lemmatizer = nltk.stem.WordNetLemmatizer()
            lem = []
            for item, tag in nltk.pos_tag(nltk.word_tokenize(s)):
                if tag.startswith("NN"):
                    try:
                        lem.append(lemmatizer.lemmatize(item, pos='n'))
                    except RecursionError:
                        lem.append('Big_word')
                elif tag.startswith('VB'):
                    try:
                        lem.append(lemmatizer.lemmatize(item, pos='v'))
                    except RecursionError:
                        lem.append('Big_word')
                elif tag.startswith('JJ'):
                    try:
                        lem.append(lemmatizer.lemmatize(item, pos='a'))
                    except RecursionError:
                        lem.append('Big_word')
                elif tag.startswith('R'):
                    try:
                        lem.append(lemmatizer.lemmatize(item, pos='r'))
                    except RecursionError:
                        lem.append('Big_word')
                else:
                    try:
                        lem.append(lemmatizer.lemmatize(item))
                    except RecursionError:
                        lem.append('Big_word')
            return lem
    else:
        def tokenizer(s):
            try:
                return nltk.word_tokenize(s)
            except TypeError:
                return ["UNKNOWN"]

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

    all_text = train["comment_text"].tolist() + test["comment_text"].tolist()
    whole = vec.fit_transform(all_text)
    train = vec.transform(train["comment_text"])
    test = vec.transform(test["comment_text"])

    if debug:
        print("Removing these tokens:\n{}".format(vec.stop_words_))

    return train, test, whole


def get_sparse_matrix(train=None, test=None, params=None, remove_numbers_function=True, debug=True, save=False,
                      load=True, data_dir="data", stemming=True, lemmatization=False):
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
            raise ValueError("You asked to load the features but they were not found"
                             + "at the specified location: \n{}\n{}".format(name_train, name_test))
    else:
        print('Computing the sparse matrixes, this will take a while...!')
        train, test, _ = tf_idf(train, test, params, remove_numbers_function, debug, stemming, lemmatization)

    if save:
        print('Saving train file as {}'.format(name_train))
        save_sparse_csr(name_train, train)
        print('Saving test file as {}'.format(name_test))
        save_sparse_csr(name_test, test)

    return train, test


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train, test = get_sparse_matrix(train, test, params=None, remove_numbers_function=True, debug=True, save=True, load=False)
