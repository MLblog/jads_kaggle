import time
import matplotlib.pyplot as plt
import numpy as np
import itertools


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


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          name=None,
                          cmap=plt.cm.Blues,
                          save_to=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    cm : array
         Object created using the function sklearn.metrics.confusion_matrix()

    normalize : bool, optional (default=False)
        If ``False``, return the cofusion matrix.
        Otherwise, return the normalized confusion matrix.

    title: str
        The title of the graph

    cmap : colormap instance, (default = plt.cm.Blues)

    name:str
        Possible name to used in the title

    save_to: str, default None
            File path to the folder that we would like to save the results
            By default all the figures are save in .eps format

    Returns
    -------
    A matplotlib object

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if name:
        plt.title(title + ' for the {}'.format(name))
    else:
        plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_to:
        plt.savefig(save_to, format='eps', bbox_inches="tight")
    plt.show()
