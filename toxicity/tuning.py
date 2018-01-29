import multiprocessing
from itertools import product
from collections import Mapping
from functools import partial

import sys

sys.path.append('..')
from utils import timing, TAGS

TUNING_OUTPUT_DEFAULT = 'data/tuning.txt'


def eval_permutation(params, predictor_cls, train_x, train_ys, method='split', nfolds=3, silent=True):
    """
    Evaluathes a predictor using a certain param set on the given training set.
    Note: This could be nested but multiprocessing can not picke it so it shall remain global.

    :param params: Parameters to be evaluated
    :param predictor_cls: The predictors class name - NOT an object of the class
    :param train_x: Contains the preprocessed input features
    :param train_ys: Dictionary mapping tag names to their array of values
    :param method: Method to be used for evaluation. Set to split for speed by default, CV might be more robust
    :param nfolds: Number of folds to be used by cross-validation (only used if method='CV')
    :param silent: Whether or not progress messages will be printed
    :return: Tuple of (params, score)
    """
    if not silent:
        print("Evaluating {}".format(params))
        sys.stdout.flush()  # Force child processes to print

    predictor = predictor_cls(**params)
    score = predictor.evaluate(train_x, train_ys, method=method, nfolds=nfolds)
    return tuple(sorted(params.items())), score


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

    permutations = get_permutations(param_grid)
    print("Applying GridSearch for {} permutations of parameters".format(len(permutations)))

    processes = min(max(1, multiprocessing.cpu_count() - 1), len(permutations))
    if not silent:
        print("Running tune in parallel using {} child processes".format(processes))

    pool = multiprocessing.Pool(processes=processes)
    evaluator = partial(eval_permutation,
                        predictor_cls=predictor_cls,
                        train_x=train_x,
                        train_ys=train_ys,
                        method=method,
                        nfolds=nfolds,
                        silent=silent)
    scores = pool.map(evaluator, permutations)

    if persist:
        # Append results to file.
        with open(write_to, "a") as f:
            f.write("------------------------------------------------\n")
            f.write("Model\t{}\n".format(predictor_cls.name))
            for params, score in scores:
                f.write("\tScore\t{}\nparams: {}\n\n".format(score, dict(params)))

    best_params, best_score = min(scores, key=lambda t: t[1])

    return dict(best_params), best_score
