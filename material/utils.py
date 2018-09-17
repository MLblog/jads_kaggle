import os
import numpy as np
from PIL import Image
from fnmatch import fnmatch


def remove_low_resolution_images(path, min_resolution=20):
    """
    A function that removes all the single pixel images

    Parameters
    -------------------------
    path: str
        Path to the folder that we would like to clear out.
    min_resolution: int, optional
        The minimum image resolution that we want to clear out of the folder.
    """
    if not isinstance(min_resolution, int):
        raise ValueError('The min resolution should be integer')
    print('I am working on the {} directory'.format(path))

    count = 0
    pattern = "*.jpg"
    for dir_path, _, files in os.walk(path):
        for name in files:
            temp_name = os.path.join(dir_path, name)
            temp_shape = np.shape(np.array(Image.open(temp_name)))
            if (temp_shape[0] < min_resolution or temp_shape[1] < min_resolution) and fnmatch(temp_name, pattern):
                os.remove(temp_name)
                count += 1
            else:
                pass
    print('I am done here and I removed {} files'.format(count))
