import time
import dill as pickle


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


def progress(text, verbose=True, same_line=False, newline_end=True):
    if verbose:
        print("{}[{}] {}".format("\r" if same_line else "", time.strftime("%Y-%m-%d %H:%M:%S"),
                                 text), end="\n" if newline_end else "")


def save_object(save_path, object_):
    """To save an object on the save path.

    Parameters
    ----------
    save_path: str
        path to save an object
    object_: object
        object to be saved
    """
    with open(save_path, 'wb') as handle:
        try:
            pickle.dump(object_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except AttributeError:
            pickle.dump(pickle.dumps(object_), handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(load_path):
    """To load a object from the save path.

    Parameters
    ----------
    load_path: str
        path to load an object
    """
    with open(load_path, 'rb') as handle:
        return pickle.load(handle)
