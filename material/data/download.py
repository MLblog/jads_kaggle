import os
import multiprocessing
import urllib3
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import json

# Set to True to print failed downloads.
DEBUG = False

# Global to reduce expensive constructor calls.
pool_manager = urllib3.PoolManager()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _parse(data_file):
    """Parse the JSON file and create a mapping from output image file paths to their url.

    Parameters
    ----------
    data_file : str
        Path to the JSON file containing the image metadata (id, url and potentially label)

    Returns
    -------
    list of (str, str)
        Pairs of image paths and urls.
    """

    # Create the label mapping ('ImageID' : 'label')
    labels = {}
    try:
        _ann = json.load(open(data_file))['annotations']
        for a in _ann:
            labels[a['image_id']] = a['label_id']
    except KeyError:
        # This JSON file does not include the 'annotations' field - it is probably a test file.
        pass

    # Parse the URLs.
    key_url_list = []
    images = json.load(open(data_file))['images']
    for item in images:
        url = item['url']
        if not isinstance(url, str):
            # The URL can either be a string, or a list of exactly 1 string.
            assert len(url) == 1
            url = url[0]

        image_id = item['image_id']
        if image_id not in labels:
            path = str(image_id)
        else:
            # Save the image in its label's specific folder.
            path = os.path.join(str(labels[image_id]), str(image_id))
        key_url_list.append((path, url))

    return key_url_list


def _create_dirs(root, filenames):
    """Create one folder for each label in the root dir. """
    for filename in filenames:
        path = os.path.join(root, str(filename))
        if not os.path.exists(path):
            os.makedirs(path)


def _dl_image(key_url):
    """Downloads an image to the specified location.

    Parameters
    ----------
    tuple of (str, str)
        The first element is the path where the image will be saved.
        The second element is the URL where the image can be accessed.

    """
    path, url = key_url
    if os.path.exists(path):
        print('Image {} already exists. Skipping download.'.format(path))
        return

    try:
        image_data = pool_manager.request('GET', url, timeout=10.0).data
        pil_image = Image.open(BytesIO(image_data))
        pil_image_rgb = pil_image.convert('RGB')
        pil_image_rgb.save(path + ".jpg", format='JPEG', quality=90)

    except:  # noqa
        if DEBUG:
            print("Failed to Download image {}".format(path))


def dl_images(data_file, out_dir, processes=20):
    """Download all images found in the metadata file to the specified folder.

    Parameters
    ----------
    data_file : str
        Path to the JSON file containing the image metadata (id, url and potentially label)
    out_dir : str
        Path to the folder where the images will be saved.
    processes : int, optional
        Number of OS processes to be used. Increasing this number may speed-up the process but will also
        use more RAM and CPU.

    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = _parse(data_file)

    # If the label exists, then each key is in the form "label/imageId". Else it just "imageID".
    if "/" in key_url_list[0][0]:
        # Create a folder for each unique label to better organize images.
        labels = set([entry[0].split("/")[0] for entry in key_url_list])
        _create_dirs(out_dir, labels)
    elif "\\" in key_url_list[0][0]:
        # The same for Windows paths.
        labels = set([entry[0].split("\\")[0] for entry in key_url_list])
        _create_dirs(out_dir, labels)

    # Complement the relatives image paths with their root.
    key_url_list = [(os.path.join(out_dir, path), url) for (path, url) in key_url_list]

    pool = multiprocessing.Pool(processes=processes)

    with tqdm(total=len(key_url_list)) as t:
        for _ in pool.imap_unordered(_dl_image, key_url_list):
            t.update(1)
