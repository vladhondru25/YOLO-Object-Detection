import numpy as np

import torch
from torchvision import transforms, utils


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, image, bboxes):
        image = transforms.ToTensor()(image)
        bb_targets = transforms.ToTensor()(bboxes)

        return image, bb_targets