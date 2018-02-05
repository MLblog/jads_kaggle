import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import os
from utils import timing, TAGS
from preprocessing import check_compatibility

nltk.download('stopwords')
eng_stopwords = set(stopwords.words("english"))

# Input dataframes are assumed to contain plain text in this column
TEXT_COLUMN = "comment_text"


class FeatureAdder(object):
    def __init__(self, data_dir="data", upper_case=False, word_count=False, unique_words_count=False,
                 letter_count=False, punctuation_count=False, little_case=False,
                 stopwords=False, question_or_exclamation=False, number_bad_words=False):

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
            self._count_bad_words: number_bad_words
        }

    def set_path(self, data_dir):
        self.data_dir = data_dir

    def _count_bad_words(self, df):
        """
        This is a method that creates a new feature with the number of words in the google bad list.
        Source: https://www.freewebheaders.com/full-list-of-bad-words-banned-by-google/
        Parameters
        -------------------------
        df: Pandas Dataframe. Assumed to contain
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
        df["count_bad_words"] = df[TEXT_COLUMN].apply(count_badwords)
        return df

    def _upper(self, df):
        """
        This is a method that creates a new feature with the number of capitalized words.

        Parameters
        -------------------------
        df: pd.Dataframe. Assumed to contain text in a column named `TEXT_COLUMN`

        Returns
        --------------------------
        pd.Dataframe with the number of the capitalized words as an extra feature.
        """
        df["count_words_upper"] = df[TEXT_COLUMN].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        return df

    def _count_words(self, df):
        """
        This is a method that creates a new feature with the number of words in the sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `TEXT_COLUMN`

        Returns
        --------------------------
        pd.Dataframe with the number of words as an extra feature.
        """
        df['count_word'] = df[TEXT_COLUMN].apply(lambda x: len(str(x).split()))
        return df

    def _unique_words(self, df):
        """
        This is a method that creates a new feature with the number of unique words in the sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `TEXT_COLUMN`

        Returns
        --------------------------
        pd.Dataframe with the number of the unique words as an extra feature.
        """
        df['count_unique_word'] = df[TEXT_COLUMN].apply(lambda x: len(set(str(x).split())))
        return df

    def _count_letters(self, df):
        """
        This is a method that creates a new feature with the number of letters in the sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `TEXT_COLUMN`

        Returns
        --------------------------
        pd.Dataframe with the aggregated number of the characters as an extra feature.
        """
        df['count_letters'] = df[TEXT_COLUMN].apply(lambda x: len(str(x)))
        return df

    def _count_punctuation(self, df):
        """
        This is a method that creates a new feature with the number of the punctuation marks.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `TEXT_COLUMN`

        Returns
        --------------------------
        pd.Dataframe with the number of the puncutation symbols as an extra feature.
        """
        df["count_punctuations"] = df[TEXT_COLUMN].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation]))
        return df

    def _count_little_case(self, df):
        """
        This is a method that creates a new feature with the number of the lowercase words.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `TEXT_COLUMN`

        Returns
        --------------------------
        pd.Dataframe with the number of the not capitalized words as an extra feature.
        """
        df["count_words_title"] = df[TEXT_COLUMN].apply(lambda x: len([w for w in str(x).split() if w.islower()]))
        return df

    def _count_stopwords(self, df):
        """
        This is a method that creates a new feature with the number of the stop words in a sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `TEXT_COLUMN`

        Returns
        --------------------------
        pd.Dataframe with the number of the stop words (like 'then', 'to', 'a' etc) as an extra feature.
        """
        df["count_stopwords"] = df[TEXT_COLUMN].apply(
            lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
        return df

    def _question_or_exclamation(self, df):
        """
        This is a method that creates a new feature with the number of the question or exclamation marks used in the sentence.

        Parameters
        -------------------------
        df: pd.Dataframe, assumed to contain text in a column named `TEXT_COLUMN`

        Returns
        --------------------------
        pd.Dataframe with the number of the question or exclamation marks as an extra feature.
        """
        df['question_mark'] = df[TEXT_COLUMN].str.count('\?')
        df['exclamation_mark'] = df[TEXT_COLUMN].str.count('!')
        return df

    @check_compatibility
    @timing
    def add_features(self, train, test):
        """
        Call feature extractors that have been activated (by setting their boolean attribute to True)

        Parameters
        -------------------------
        train, test: pd.Dataframes to go through the transformation

        Returns
        --------------------------
        (train_x, test_x): Matrices containing all features

        Example
        --------------------------
            >>> fa = FeatureAdder(upper_case=True, word_count=True, unique_words_count=True)  # Activate desired feature extraction
            >>> train, test = fa.add_features(train, test)
        """
        for method, condition in self.features.items():
            if condition:
                method(train), method(test)

        train_x = train.drop(TAGS + ['id', TEXT_COLUMN], axis=1).as_matrix()
        test_x = test.drop(['id', TEXT_COLUMN], axis=1).as_matrix()
        return train_x, test_x


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    da = FeatureAdder(upper_case=True, word_count=True, unique_words_count=True,
                      letter_count=True, punctuation_count=True, little_case=True,
                      stopwords=True, question_or_exclamation=True, number_bad_words=True)

    df_train, df_test = da.add_features(train, test)
    print(df_train.head(1))
