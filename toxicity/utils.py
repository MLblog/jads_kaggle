import time
import pandas as pd
from itertools import product
from collections import Mapping

from utils import timing

TAGS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
TUNING_OUTPUT_DEFAULT = 'tuning.txt'


def get_permutations(d):
    """
    Finds all possible permutations found in the input dictionary

    :param d: A mapping from a parameter's to a list of its valid values
    :return: List of possible permutation

    Example:
        >>> d = {'A': [1, 2], 'B': [3, 4]}
        >>> get_permutations(d) == [{'A': 1, 'B': 3}, {'A': 2, 'B': 3}, {'A': 1, 'B': 4}, {'A': 2, 'B': 4}]
        >>> True
    """
    permutations = []

    if isinstance(d, Mapping):
        # wrap dictionary in a singleton list to support either dict
        # or list of dicts
        d = [d]

    for p in d:
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(p.items())
        if not p.items():
            permutations.append({})
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                permutations.append(params)

    return permutations


@timing
def tune(predictor_cls, train_x, train_ys, param_grid, method='split', nfolds=3, silent=True, persist=True,
         write_to=TUNING_OUTPUT_DEFAULT):
    """
    Exhaustively searches over the grid of parameters for the best combination by minimizing the log loss.

    :param predictor_cls: The predictors class name - NOT an object of the class
    :param train_x Contains the preprocessed input features
    :param train_ys Dictionary mapping tag names to their array of values
    :param param_grid: Grid of parameters to be explored.
    :param method: Method to be used for evaluation. Set to split for speed by default, CV might be more robust
    :param nfolds: Number of folds to be used by cross-validation (only used if method='CV')
    :param silent: Whether or not progress messages will be printed
    :param persist: If set to true, will write tuning results to a file
    :param write_to: If persist is set to True, write_to defines the filepath to write to
    :return: tuple of: (Best parameters found, Best score achieved).
    """

    permutations = get_permutations(param_grid)
    print("Applying GridSearch for {} permutations of parameters".format(len(permutations)))

    # TODO: Parallelize this loop
    best_score, best_params = 1, {}
    for params in permutations:
        if not silent:
            print("Evaluating {}".format(params))
        predictor = predictor_cls(**params)
        score = predictor.evaluate(train_x, train_ys, method=method, nfolds=nfolds)
        if score < best_score:
            best_score = score
            best_params = params

    if persist:
        # Append results to file.
        with open(write_to, "a") as f:
            f.write("------------------------------------------------\n")
            f.write("Model\t{}\n".format(predictor.name))
            f.write("Best Score\t{}\nparams: {}\n\n".format(best_score, best_params))

    return best_params, best_score


def create_submission(predictor, train_x, train_ys, test_x, test_id, write_to):
    """
    Creates a submissions file for the given test set

    :param predictor: The predictor to be used for fitting and predicting
    :param train_x: The (preprocessed) features to be used for fitting
    :param train_ys: A dictionary from tag name to its values in the training set.
    :param test_x: The (preprocessed) features to be used for predicting.
    :param write_to: A file path where the submission is written
    """

    submission = pd.DataFrame({'id': test_id})
    for tag in TAGS:
        print("{} Fitting on {} tag".format(predictor, tag))
        predictor.fit(train_x, train_ys[tag])
        submission[tag] = predictor.predict(test_x)

    submission.to_csv(write_to, index=False)
    print("Submissions created at location " + write_to)


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
