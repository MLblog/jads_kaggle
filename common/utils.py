import time


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
