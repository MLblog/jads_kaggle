import os
import random
import shutil
from abc import ABCMeta, abstractmethod


class Sampler(metaclass=ABCMeta):
    """A base abstract class of a sampler. Concrete implementations will provide strategies to
    construct a reduced dataset from the original one to quickly prototype solutions.

    """

    def conditional_create(self, path, debug=True):
        """Create directory if it does not exist.

        Parameters
        ----------
        path : str
            The path to the directory to be created.
        debug : bool
            Whether or not a debugging message should be printed to the console.

        Returns
        -------
        bool
            Whether the file was created. False if it already existed.

        """
        if not os.path.exists(path):
            if debug:
                print("Creating output directory at {}".format(path))
            os.makedirs(path)
            return True

        return False

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
        self.conditional_create(out_path)

        for label in labels:
            label = str(label)
            input = os.path.join(in_path, label)

            if not os.path.exists(input):
                raise ValueError("Could not find directory {}".format(input))

            output = os.path.join(out_path, label)
            try:
                shutil.copytree(input, output)
            except FileExistsError:
                print("file {} already exists, skipping".format(output))


class PercentageSelector(Sampler):
    """This class implements the Sampler interface by providing a strategy to reduce a dataset.

    This sampler will copy a percentage of pictures for all labels.

    Examples
    --------
    >>> in_path = "../data/train/"
    >>> out_path = "../data/sample/train"
    >>>
    >>> # Select the labels 5 and 104 out of all 128 available labels.
    >>> ps = PercentageSelector()
    >>> ps.sample(0.1, in_path, out_path)

    """
    def __init__(self):
        pass

    def sample(self, percentage, in_path, out_path):
        """Sample the dataset by selecting a percentage of pictures for all labels.

        Parameters
        ----------
        percentage: float
            The Percentage of images to be selected.
        in_path : str
            Path to the input folder. It is assumed that this will contain one sub-folder for each label. A subset of
            these sub-folders will be sampled.
        out_path : str
            Path to the output folder where the sample will be stored. If it does not exist, this folder will be
            crated.

        """
        assert percentage <= 1 and percentage > 0

        created = self.conditional_create(out_path)
        if not created:
            raise ValueError("output folder {} already exists".format(out_path))

        # Our labels are named with integers, ranging from 1 to 128.
        labels = range(1, 129)

        for label in labels:
            label = str(label)

            input = os.path.join(in_path, label)
            if not os.path.exists(input):
                raise ValueError("Could not find directory {}".format(input))

            output = os.path.join(out_path, label)
            self.conditional_create(output)

            # Read the images
            images = [i for i in os.listdir(input) if i.endswith('.jpg')]

            # Number of images to be sampled for this particular label.
            num_images = int(percentage * len(images))

            # Randomly select num_images from the image folder and copy them to the output
            rand_sample = [images[i] for i in sorted(random.sample(range(len(images)), num_images))]
            for image in rand_sample:
                shutil.copyfile(os.path.join(input, image), os.path.join(output, image))


if __name__ == "__main__":
    in_path = "data/train/"
    out_path = "data/sample/train"

    ls = LabelSelector()
    ls.sample([5, 104], in_path, out_path)
