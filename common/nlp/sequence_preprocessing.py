import re
import string
import copy


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
            "that's": "that is"}


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
                     "thats": "that is"}


def preprocess_text_for_DL(*datasets, text_col="question_text", word_map=WORD_MAP):
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

    Notes
    -----
    - Take into account that the function first lower cases all text and then performs the word mapping
    (before removing punctuation). So, in the mapping, use {"won't": "will not"} instead
    of {"Wont": "Will not}, etc.
    - The method takes an arbitrary number of datasets, so it's possible to preprocess just train or test
    or both. Just make sure all datasets have the same column name for the text column.

    Returns
    -------
    new_datasets: tuple of pd.DataFrames
        The datasets with cleaned text columns.
    """
    def clean_string(text, character_map):
        """Cleans a single string, i.e., removes punctuation and makes it lower case."""
        return text.translate(character_map).lower()

    # make translation of punctuation characters
    punctuation_map = str.maketrans('', '', string.punctuation)

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
        data[text_col] = data[text_col].apply(lambda x: clean_string(x, character_map=punctuation_map))

    return new_datasets[0] if len(new_datasets) == 1 else new_datasets
