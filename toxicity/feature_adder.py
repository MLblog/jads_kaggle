import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from utils import timing

# settings. This requires running `nltk.download("stopwords")`
nltk.download('stopwords')
eng_stopwords = set(stopwords.words("english"))

# Input dataframes are assumed to contain plain text in this column
TEXT_COLUMN = "comment_text"


class FeatureAdded(object):
    def __init__(self, upper_case=False, word_count=False, unique_words_count=False,
                 letter_count=False, punctuation_count=False, little_case=False,
                 stopwords=False, question_or_exclamation=False, number_bad_words=False):

        self.features = {
            self._upper: upper_case,
            self._count_words: word_count,
            self._unique_words: unique_words_count,
            self._count_letters: letter_count,
            self._count_punctuation: punctuation_count,
            self._count_little_case: little_case,
            self._count_stopwords: stopwords,
            self._question_or_exclamation: question_or_exclamation,
            self._count_bad_words: number_bad_words
        }

    @staticmethod
    def _count_bad_words(df):
        """
    This is a method that creates a new feature with the number of words in the google bad list.
    Source: https://www.freewebheaders.com/full-list-of-bad-words-banned-by-google/

    Parameters
    -------------------------
    df: Pandas Dataframe. Assumed to contain

    Returns
    --------------------------
    df: Data frame with the number of the bad words accoring to the google dictionary.

        """

        def union_sets(*dfs):
            final = set()
            for df in dfs:
                s = set(df[list(df)[0]])  # Assuming that curse words are the 1st column
                final = final.union(s)
            return [' {} '.format(x) for x in final]

        def count_badwords(comment):
            return sum(badword.lower() in comment.lower() for badword in badwords)

        badwords_1 = pd.read_csv("data/dictionaries/google_bad_words.csv", 'utf-8')
        badwords_2 = pd.read_csv("data/dictionaries/bad_words.csv", sep=',')

        badwords = union_sets(badwords_1, badwords_2)
        df["count_bad_words"] = df[TEXT_COLUMN].apply(count_badwords)
        return df

    @staticmethod
    def _upper(df):
        """
    This is a method that creates a new feature with the number of capitalized words.

    Parameters
    -------------------------
    df: Pandas Dataframe. Assumed to contain

    Returns
    --------------------------
    int: The number of the capitalized words.

    Example:
        >>>_upper('Mimis is such a GOOD BOY!!!')
        3
        """
        df["count_words_upper"] = df[TEXT_COLUMN].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
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
        df['count_word'] = df[TEXT_COLUMN].apply(lambda x: len(str(x).split()))
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

        df['count_unique_word'] = df[TEXT_COLUMN].apply(lambda x: len(set(str(x).split())))
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
        df['count_letters'] = df[TEXT_COLUMN].apply(lambda x: len(str(x)))
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
        df["count_punctuations"] = df[TEXT_COLUMN].apply(
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
        df["count_words_title"] = df[TEXT_COLUMN].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
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
        df["count_stopwords"] = df[TEXT_COLUMN].apply(
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
        df['question_mark'] = df[TEXT_COLUMN].str.count('\?')
        df['exclamation_mark'] = df[TEXT_COLUMN].str.count('!')
        return df

    @timing
    def add_features(self, train, test):
        for method, condition in self.features.items():
            if condition:
                method(train), method(test)

        return train, test


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    df = FeatureAdded(upper_case=True, word_count=True, unique_words_count=True,
                      letter_count=True, punctuation_count=True, little_case=True,
                      stopwords=True, question_or_exclamation=True, number_bad_words=True)

    df_train, df_test = df.add_features(train, test)
    print(df_train.head(1))
