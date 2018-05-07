import multiprocessing
from itertools import product
from collections import Mapping
from functools import partial

import sys

from GPyOpt.methods import BayesianOptimization

sys.path.append('..')
from utils import timing # noqa

TUNING_OUTPUT_DEFAULT = 'data/tuning.txt'


def eval_permutation(params, predictor_cls, train_x, train_ys, method='split', nfolds=3, silent=True):
    """
    Evaluates a predictor using a certain param set on the given training set.
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


def write_results(write_to, scores, predictor_cls):
    """ Writes experiment results to specified file """
    with open(write_to, "a") as f:
            f.write("------------------------------------------------\n")
            f.write("Model:\t{}\n".format(predictor_cls.name))
            for params, score in scores:
                f.write("Score:\t{}\nparams:\t{}\n\n".format(score, dict(params)))


def bayesian_optimization(predictor_cls, train_x, train_ys, params, max_iter, max_time=600, model_type='GP', acquisition_type='EI',
                          acquisition_weight=2, eps=1e-6, batch_method='local_penalization', batch_size=1, method='split', nfolds=3,
                          silent=True, persist=True, write_to=TUNING_OUTPUT_DEFAULT):
    """
    Automatically configures hyperparameters of ML algorithms. Suitable for reasonably small sets of params.

    :param predictor_cls: The predictors class
    :param train_x Contains the preprocessed input features
    :param train_ys Dictionary mapping tag names to their array of values
    :param params: Dictionary of parameters, type (continuous/discrete), and their allowed ranges/values.
           NOTE: param_ranges must first contain continuous variables, then discrete.
    :param max_iter: Maximum number of iterations / evaluations.
           NOTE: excluding initial exploration session, might converge earlier.
    :param max_time: Maximum time to be used in optimization.
    :param model_type: Model used for optimization. Defaults to Gaussian Process ('GP').
        - 'GP', standard Gaussian process.
        - 'GP_MCMC',  Gaussian process with prior in the hyper-parameters.
        - 'sparseGP', sparse Gaussian process.
        - 'warperdGP', warped Gaussian process.
        - 'InputWarpedGP', input warped Gaussian process.
        - 'RF', random forest (scikit-learn).
    :param acquisition_type: Function used to determine the next parameter settings to evaluate.
        - 'EI', expected improvement.
        - 'EI_MCMC', integrated expected improvement (requires GP_MCMC model).
        - 'MPI', maximum probability of improvement.
        - 'MPI_MCMC', maximum probability of improvement (requires GP_MCMC model).
        - 'LCB', GP-Lower confidence bound.
        - 'LCB_MCMC', integrated GP-Lower confidence bound (requires GP_MCMC model).
    :param acquisition_weight: Exploration vs Exploitation parameter.
    :param eps: Minimum distance between consecutive candidates x.
    :param batch_method: Determines the way the objective is evaluated if batch_size > 1 (all equivalent if batch_size=1).
        - 'sequential', sequential evaluations.
        - 'random': synchronous batch that selects the first element as in a sequential policy and the rest randomly.
        - 'local_penalization': batch method proposed in (Gonzalez et al. 2016).
        - 'thompson_sampling': batch method using Thompson sampling.
    :param batch_size: Number of parallel optimizations to run. If None, uses batch_size = number of cores.
    :param method: Method to be used for evaluation. Set to split for speed by default, CV might be more robust
    :param nfolds: Number of folds to be used by cross-validation (only used if method='CV')
    :param silent: Whether or not progress messages will be printed
    :param persist: If set to true, will write tuning results to a file
    :param write_to: If persist is set to True, write_to defines the filepath to write to
    :return: tuple of: (Best parameters found, Best score achieved).
    """

    print("Applying Bayesian Optimization to configure {} in at most {} iterations and {} seconds."
          .format(predictor_cls, max_iter, max_time))

    def create_mapping(p_array):
        """ Changes the 2d np.array from GPyOpt to a dictionary. """
        mapping = dict()
        for i in range(len(params)):
            mapping[params[i]["name"]] = p_array[0, i]

        return mapping

    # define the optimization function
    def f(parameter_array):
        param_dict = create_mapping(parameter_array)
        score = eval_permutation(params=param_dict,
                                 predictor_cls=predictor_cls,
                                 train_x=train_x,
                                 train_ys=train_ys,
                                 method=method,
                                 nfolds=nfolds,
                                 silent=silent)

        scores.append(score)
        # only return score to optimizer
        return score[1]

    # scores are added to this list in the optimization function f
    scores = []

    # run optimization in parallel
    num_cores = max(1, multiprocessing.cpu_count() - 1)

    # set batch_size equal to num_cores if no batch_size is provided
    if not batch_size:
        batch_size = num_cores

    if not silent:
        print("Running Bayesian Optimization in batches of {} on {} cores using {}.".format(batch_size, num_cores, batch_method))

    # define optimization problem
    opt = BayesianOptimization(f, domain=params, model_type=model_type, acquisition_type=acquisition_type,
                               normalize_Y=False, acquisition_weight=acquisition_weight, num_cores=num_cores, batch_size=batch_size)

    # run optimization
    opt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=False)

    # report results
    if persist:
        write_results(write_to, scores, predictor_cls)

    best_params, best_score = max(scores, key=lambda t: t[1])

    return dict(best_params), best_score


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
        write_results(write_to, scores, predictor_cls)

    best_params, best_score = max(scores, key=lambda t: t[1])

    return dict(best_params), best_score
