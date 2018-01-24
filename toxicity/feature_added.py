import pandas as pd
from nltk.corpus import stopwords
import string

# settings
eng_stopwords = set(stopwords.words("english"))


class FeatureAdded(object):
    def __init__(self, upper_case=False, word_count=False, unique_words_count=False,
                 letter_count=False, punctuation_count=False, little_case=False,
                 stopwords=False, question_or_exclamation=False):
        self.upper_case = upper_case
        self.word_count = word_count
        self.unique_words_count = unique_words_count
        self.letter_count = letter_count
        self.punctuation_count = punctuation_count
        self.little_case = little_case
        self.stopwords = stopwords
        self.question_or_exclamation = question_or_exclamation

    @staticmethod
    def _upper(df):
        """
    This is a method that creates a new feature with the number of capitalized words.

    Parameters
    -------------------------
    df: Pandas Dataframe

    Returns
    --------------------------
    int: The number of the capitalized words.

    Example:
        >>>_upper('Mimis is such a GOOD BOY!!!')
        3
        """
        df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        return df

    @staticmethod
    def _count_words(df):
        """
    This is a method that creates a new feature with the number of words in the sentence.

    Parameters
    -------------------------
    df: Pandas Dataframe

    Returns
    --------------------------
    int: The number of the  words.

    Example:
        >>>_count_words('Mimis is such a GOOD BOY!!!')
        6
        """
        df['count_word'] = df["comment_text"].apply(lambda x: len(str(x).split()))
        return df

    @staticmethod
    def _unique_words(df):
        """
    This is a method that creates a new feature with the number of unique words in the sentence.

    Parameters
    -------------------------
    df: Pandas Dataframe

    Returns
    --------------------------
    int: The number of the  unique words.

    Example:
        >>>_unique_words('Mimis is such a GOOD BOY. Hello Mimis!')
        7
        """

        df['count_unique_word'] = df["comment_text"].apply(lambda x: len(set(str(x).split())))
        return df

    @staticmethod
    def _count_letters(df):
        """
    This is a method that creates a new feature with the number of letters in the sentence.

    Parameters
    -------------------------
    df: Pandas Dataframe

    Returns
    --------------------------
    int: The number of the  words.

    Example:
        >>>_count_letters('Mimis is such a GOOD BOY!!!')
        27
        """
        df['count_letters'] = df["comment_text"].apply(lambda x: len(str(x)))
        return df

    @staticmethod
    def _count_punctuation(df):
        """
    This is a method that creates a new feature with the number of the punctuation marks.

    Parameters
    -------------------------
    df: Pandas Dataframe

    Returns
    --------------------------
    int: The number of the marks.

    Example:
    >>>_count_punctuation('Mimis is such a GOOD BOY!!!')
    3
    """
        df["count_punctuations"] = df["comment_text"].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation]))
        return df

    @staticmethod
    def _count_little_case(df):
        """
    This is a method that creates a new feature with the number of the lowercase words.

    Parameters
    -------------------------
    df: Pandas Dataframe

    Returns
    --------------------------
    int: The number of the  words.

    Example:
    >>>_count_little_case('Mimis is such a GOOD BOY!!!')
    2
    """
        df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
        return df

    @staticmethod
    def _count_stopwords(df):
        """
    This is a method that creates a new feature with the number of the words in a a sentence without the stopwords.

    Parameters
    -------------------------
    df: Pandas Dataframe

    Returns
    --------------------------
    int: The number of the  words.

    Example:
    >>>_count_stopwords('Mimis is such a GOOD BOY!!!')
    3
        """
        df["count_stopwords"] = df["comment_text"].apply(
            lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
        return df

    @staticmethod
    def _question_or_exclamation(df):
        """
    This is a method that creates a new feature with the number of the question or exclamation marks used in the sentence.

    Parameters
    -------------------------
    df: Pandas Dataframe

    Returns
    --------------------------
    int: The number of the marks.
        """
        df['question_mark'] = df['comment_text'].str.count('\?')
        df['exclamation_mark'] = df['comment_text'].str.count('!')
        return df

    def add_features(self, train, test):
        if self.upper_case:
            self._upper(train), self._upper(test)
        if self.word_count:
            self._count_words(train), self._count_words(test)
        if self.unique_words_count:
            self._unique_words(train), self._unique_words(test)
        if self.letter_count:
            self._count_letters(train), self._count_letters(test)
        if self.punctuation_count:
            self._count_punctuation(train), self._count_punctuation(test)
        if self.little_case:
            self._count_little_case(train), self._count_little_case(test)
        if self.stopwords:
            self._count_stopwords(train), self._count_stopwords(test)
        if self.question_or_exclamation:
            self._question_or_exclamation(train), self._question_or_exclamation(test)
        return train, test


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    df = FeatureAdded(upper_case=True, word_count=True, unique_words_count=True,
                      letter_count=True, punctuation_count=True, little_case=True,
                      stopwords=True, question_or_exclamation=True)

    df_train, df_test = df.add_features(train, test)
    print(df_train.head(1))
