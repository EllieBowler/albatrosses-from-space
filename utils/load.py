import numpy as np
from .utils import interp_method_cv2
import cv2


def load_image_vgg16(path, method=cv2.INTER_LINEAR):

    # Take mean and std for vgg16 and expand for extra channels
    MODEL_MEAN = np.array([0.485, 0.456, 0.406])
    MODEL_STD = np.array([0.229, 0.224, 0.225])
    MODEL_MEAN_X = np.insert(MODEL_MEAN, [0, 3], MODEL_MEAN.mean())
    MODEL_STD_X = np.insert(MODEL_STD, [0, 3], MODEL_STD.mean())

    img = np.load(path)
    pan = img['I_PAN'] / 2047.
    ms = img['I_MS'] / 2047.

    pan_channels, pan_height, pan_width = pan.shape
    ms_up = interp_method_cv2(ms, pan_height, pan_width, method)

    img = np.concatenate((pan, ms_up), axis=0)
    img = (img - MODEL_MEAN_X[:, None, None]) / MODEL_STD_X[:, None, None]
    return img.astype(np.float32)


def load_tgt_vgg16(path):
    """
    Load the binary target map, dilate points to 3x3 squares
    :param path: (str) specifying path to image patch
    :return: binary target array
    """

    # Load image, need to remove channels before using dilate
    tgt = np.load(path)
    tgt = tgt.squeeze()

    # Dilate to 3x3 targets
    kernel = np.ones((3, 3), np.float)
    tgt = cv2.dilate(tgt.astype(float), kernel)

    # Add channels axis back before returning
    return tgt[None, :, :]


class AbstractImageAccessor(object):
    def get_path(self, index):
        raise NotImplementedError

    def __getitem__(self, item):
        """
        Get images identified by item
        item can be:
        - an index as an integer
        - an array of indices
        """
        if isinstance(item, int):
            # item is an integer, get a single item
            path = self.get_path(item)
            img = self.load_fn(path)
            return img
        elif isinstance(item, np.ndarray):
            # item is an array of indices
            # Get the paths of the images in the mini-batch
            paths = [self.get_path(i) for i in item]
            # Load each image
            images = [self.load_fn(path) for path in paths]
            # Stack in axis 0 to make an array of shape (sample, channel, height, width)
            return np.stack(images, axis=0)


class ImageAccessor(AbstractImageAccessor):
    def __init__(self, paths, load_fn):
        """
        Constructor
        paths - the list of paths of the images that we are to access
        """
        self.paths = paths
        self.load_fn = load_fn

    def __len__(self):
        """
        The length of this array
        """
        return len(self.paths)

    def get_path(self, index):
        return self.paths[index]


class RandomGTAccessor(AbstractImageAccessor):
    def __init__(self, paths_patch_obs, load_fn):
        """
        Constructor
        paths - the list of paths of the images that we are to access
        """
        # Flip the structure from (obs, patch) -> (patch, obs)
        self.paths_patch_obs = list(zip(*paths_patch_obs))
        self.load_fn = load_fn

    def __len__(self):
        """
        The length of this array
        """
        return len(self.paths_patch_obs)

    def get_path(self, index):
        obs_paths = self.paths_patch_obs[index]
        return np.random.choice(obs_paths)