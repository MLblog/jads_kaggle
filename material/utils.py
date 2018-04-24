import os
import numpy as np
from PIL import Image
from fnmatch import fnmatch


def clear_folder(path, unwanted_resolution=(1, 1, 3)):
    """
    A function that removes all the single pixel images

    Parameters
    -------------------------
    path: str
        Path to the folder that we would like to clear out.
    unwanted_resolution: tuple, optional
        The image resolution that we want to clear out of the folder.
    """
    print('I am working on the {} folder'.format(path))
    count = 0
    pattern = "*.jpg"
    for dir_path, _, files in os.walk(path):
        for name in files:
            temp_name = os.path.join(dir_path, name)
            temp_shape = np.shape(np.array(Image.open(temp_name)))
            if temp_shape == unwanted_resolution and fnmatch(temp_name, pattern):
                os.remove(temp_name)
                count += 1
            else:
                pass
    print('I am done here and I removed {} files'.format(count))
