# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
from PIL import Image

from albumentations import *
from albumentations.core.transforms_interface import ImageOnlyTransform

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

MEAN = [0.4984]
SD = [0.2483]


def preprocess_imagenet(image, channels=3, height=224, width=224):
    """Pre-processing for Imagenet-based Image Classification Models:
        resnet50, vgg16, mobilenet, etc. (Doesn't seem to work for Inception)

    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 224 for Imagenet data)
    width: int
        The desired width of the image  (usually 224 for Imagenet data)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logger.debug(
            "Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    assert img_data.shape[0] == channels

    for i in range(img_data.shape[0]):
        # Scale each pixel to [0, 1] and normalize per channel.
        img_data[i, :, :] = (img_data[i, :, :] / 255 -
                             mean_vec[i]) / stddev_vec[i]

    return img_data


def preprocess_inception(image, channels=3, height=224, width=224):
    """Pre-processing for InceptionV1. Inception expects different pre-processing
    than {resnet50, vgg16, mobilenet}. This may not be totally correct,
    but it worked for some simple test images.

    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 224 for Imagenet data)
    width: int
        The desired width of the image  (usually 224 for Imagenet data)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.BILINEAR)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logger.debug(
            "Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])

    return img_data


def preprocess_pylon(image, channels=1, height=256, width=256):
    img = np.array(image.convert('L'))
    transform = make_transform('eval', 256)
    img = transform(image=img)['image']
    return img


def make_transform(
        augment,
        size=256,
        rotate=90,
        p_rotate=0.5,
        brightness=0.5,
        contrast=0.5,
        min_size=0.7,
        interpolation='cubic',
):
    """Preprocess an image before passing it through a Pylon model
    This version of codes is adapted from https://github.com/cmb-chula/pylon
    for a newest version, clone the repo an from pylon.dataset import make_transform
    Args:
        augment (str): Type of augmentation (e.g. `common`, `eval`)
        size (int, optional): Size of the transformed image. Defaults to 256.
        rotate (int, optional): Degree of rotation for image augmentation. Defaults to 90.
        p_rotate (float, optional): Probability of applying the Rotation transform. Defaults to 0.5.
        brightness (float, optional): Brightness for Brightness transformation. Defaults to 0.5.
        contrast (float, optional): Contrast for Contrast transformation. Defaults to 0.5.
        min_size (float, optional): Minimum size for RandomResizedCrop transformation. Defaults to 0.7.
        interpolation (str, optional): Interpotation for Resize transform. Defaults to 'cubic'.
    Raises:
        NotImplementedError: When `augment` does not match any existing implementation
    Returns:
        function: an albumentations' Transform function
    """

    inter_opts = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
    }
    inter = inter_opts[interpolation]

    trans = []
    trans += [
        Resize(size, size, interpolation=inter),
        Normalize(MEAN, SD),
    ]

    trans += [GrayToArray()]
    return Compose(trans)


class GrayToArray(ImageOnlyTransform):
    def apply(self, img, **params):
        # because of gray scale
        # we add an additional channel
        return np.expand_dims(img, axis=0)
