import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, box, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
        # Update box coordinates
        box[2] += padw  # x_max
        box[3] += padh  # y_max
    return img, box


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, box):
        for t in self.transforms:
            image, target, box = t(image, target, box)
        return image, target, box


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target, box):
        size = random.randint(self.min_size, self.max_size)
        old_w, old_h = image.size
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        new_w, new_h = image.size
        # Scale box coordinates
        scale_x = new_w / old_w
        scale_y = new_h / old_h
        box = box * [scale_x, scale_y, scale_x, scale_y]
        return image, target, box

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, box):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            w = image.size[0]
            # Adjust box for horizontal flip
            box = [w - box[2], box[1], w - box[0], box[3]]
        return image, target, box


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, box):
        image, box = pad_if_smaller(image, self.size, box)
        target, _ = pad_if_smaller(target, self.size, box, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        i, j, h, w = crop_params
        image = F.crop(image, i, j, h, w)
        target = F.crop(target, i, j, h, w)
        box = [max(0,box[0] - j), max(0,box[1] - i), min(box[2] - j,image.size[0]), min(box[3] - i,image.size[1])]
        return image, target, box



class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, box):
        # Center crop image and target
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        
        # Get cropped image dimensions
        w, h = image.size
        
        # Adjust box to ensure it fits within the cropped image
        x_min, y_min, x_max, y_max = box
        x_min = max(0, min(x_min, w))
        y_min = max(0, min(y_min, h))
        x_max = max(0, min(x_max, w))
        y_max = max(0, min(y_max, h))
        
        # Handle cases where box becomes invalid due to crop (e.g., box area is 0)
        if x_min >= x_max or y_min >= y_max:
            box = [0, 0, 0, 0]  # Set to an invalid box or handle as per use case
        
        box = [x_min, y_min, x_max, y_max]
        return image, target, box



class ToTensor(object):
    def __call__(self, image, target, box):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        box = torch.as_tensor(box, dtype=torch.float32)
        return image, target, box


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, box):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, box


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target, box):
        image = T.ColorJitter(brightness=self.brightness,
                              contrast=self.contrast,
                              saturation=self.saturation,
                              hue=self.hue)(image)
        return image, target, box


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, box):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
            h = image.size[1]
            # Adjust box for vertical flip
            box = [box[0], h - box[3], box[2], h - box[1]]
        return image, target, box