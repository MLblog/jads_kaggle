import os
import shutil
from abc import ABCMeta, abstractmethod


class Sampler(metaclass=ABCMeta):
    """A base abstract class of a sampler. Concrete implementations will provide strategies to
    construct a reduced dataset from the original one to quickly prototype solutions.

    """

    @abstractmethod
    def sample(self):
        """Concrete implementations **must** override this method to sample a dataset. """
        pass


class LabelSelector(Sampler):
    """This class implements the Sampler interface by providing a strategy to reduce a dataset.

    This sampler will choose specific labels from the training and validation sets.

    Examples
    --------
    >>> in_path = "../data/train/"
    >>> out_path = "../data/sample/train"
    >>>
    >>> # Select the labels 5 and 104 out of all 128 available labels.
    >>> ls = LabelSelector()
    >>> ls.sample([5, 104], in_path, out_path)

    """

    def __init__(self):
        pass

    def sample(self, labels, in_path, out_path):
        """Sample the dataset by only selecting specific labels.

        Parameters
        ----------
        labels : list of str
            The labels to be selected.
        in_path : str
            Path to the input folder. It is assumed that this will contain one sub-folder for each label. A subset of
            these sub-folders will be sampled.
        out_path : str
            Path to the output folder where the sample will be stored. If it does not exist, this folder will be
            crated.

        """
        if not os.path.exists(out_path):
            print("Creating output directory at {}".format(out_path))
            os.makedirs(out_path)

        for label in labels:
            label = str(label)
            input = os.path.join(in_path, label)

            if not os.path.exists(input):
                raise ValueError("Could not find directory {}".format(input))

            output = os.path.join(out_path, label)
            shutil.copytree(input, output)


if __name__ == "__main__":
    in_path = "data/train/"
    out_path = "data/sample/train"

    ls = LabelSelector()
    ls.sample([5, 104], in_path, out_path)
