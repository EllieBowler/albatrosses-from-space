import numpy as np
import cv2
import os
import torch


def interp_method_cv2(mult, imheight, imwidth, method):
    """
    Takes multispectral array and interpolates to match pan dimensions
    :param mult: 3D MS array (channels, rows, cols)
    :param imheight: (int) height of the PAN array
    :param imwidth: (int) width of the PAN array
    :param method: specifies interpolation method:
                Nearest Neighbour: cv2.INTER_NEAREST
                Bi-linear: cv2.INTER_LINEAR
                Bi-cubic: cv2.INTER_CUBIC
    :return: Upsampled MS array (channels, imheight, imwidth)
    """

    channels = mult.shape[0]
    mult_up = np.zeros((channels, imheight, imwidth))

    for i in range(channels):
        mult_up[i, :, :] = cv2.resize(mult[i, :, :], dsize=(imheight, imwidth), interpolation=method)
    return mult_up.astype(mult.dtype)


def augment_batch(batch_X, batch_y):
    """
    Augment images using flips around the lines of symmetry
    :param batch_X: The input images
    :param batch_y: The target dataset
    :return: Inputs and targets randomly flipped
    """
    # n_samples = len(batch_X)

    n_samples, n_channels, imheight, imwidth = batch_X.shape

    # Augment by random crops
    cropheight, cropwidth = (412, 412)
    x = np.random.randint(0, imwidth - cropwidth)
    y = np.random.randint(0, imheight - cropheight)
    batch_X = batch_X[:, :, y:y+cropheight, x:x+cropwidth]
    batch_y = batch_y[:, :, y:y+cropheight, x:x+cropwidth]

    flip_h = np.random.randint(low=0, high=2, size=(n_samples,)) != 0
    flip_v = np.random.randint(low=0, high=2, size=(n_samples,)) != 0
    flip_hv = np.random.randint(low=0, high=2, size=(n_samples,)) != 0
    flip_vh = np.random.randint(low=0, high=2, size=(n_samples,)) != 0

    batch_X = batch_X.copy()
    batch_y = batch_y.copy()

    batch_X[flip_h, ...] = batch_X[flip_h, ...][:, :, :, ::-1]
    batch_X[flip_v, ...] = batch_X[flip_v, ...][:, :, ::-1, :]
    batch_X[flip_hv, ...] = batch_X[flip_hv, ...].transpose((0, 1, 3, 2))
    batch_X[flip_vh, ...] = batch_X[flip_vh, ...][:, :, ::-1, :].transpose((0, 1, 3, 2))

    batch_y[flip_h, ...] = batch_y[flip_h, ...][:, :, :, ::-1]
    batch_y[flip_v, ...] = batch_y[flip_v, ...][:, :, ::-1, :]
    batch_y[flip_hv, ...] = batch_y[flip_hv, ...].transpose((0, 1, 3, 2))
    batch_y[flip_vh, ...] = batch_y[flip_vh, ...][:, :, ::-1, :].transpose((0, 1, 3, 2))

    return batch_X, batch_y


def save_checkpoint(state, is_best, filename):
    """
    Save checkpoint if a new best is achieved
    :param state: current state of the network
    :param is_best: (bool) whether the loss has improved
    :param filename: (str) checkpoint name and folder destination
    :return: Saves the checkpoint model
    """
    if is_best:
        print("=> Saving a new best at {}".format(filename))
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")
    return


def load_checkpoint(checkpoint, model):
    """
    Load a checkpoint saved model
    :param checkpoint: (str) path to the checkpoint model
    :param model: torch model architecture (e.g unet)
    :return: Trained model
    """
    if not os.path.exists(checkpoint):
        print("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print('=> loaded checkpoint model (trained for {} epochs)'.format(epoch))
    return model

