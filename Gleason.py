import Hu.transforms as T

import torch
from torch.utils.data import Dataset
import os.path as osp
import os
from PIL import Image


class Gleason(Dataset):
    def __init__(self, imgdir, maskdir=None, train=True, val=False,
                 test=False, transforms=None, transform=None, target_transform=None):
        super(Gleason, self).__init__()
        self.imgdir = imgdir
        self.maskdir = maskdir
        self.imglist = sorted(os.listdir(imgdir))[1:]
        if not test:
            self.masklist = [item.replace('.jpg', '_classimg_nonconvex.png') for item in self.imglist]
        else:
            self.masklist = []
        self.train = train
        self.val = val
        self.test = test
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        image = Image.open(osp.join(self.imgdir, self.imglist[idx]))
        if not self.test:
            mask = Image.open(osp.join(self.maskdir, self.masklist[idx]))
            y_indices, x_indices = np.where(np.array(mask) > 0)
            if x_indices.size > 0 and y_indices.size > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                H, W = np.array(mask).shape
                x_min = max(0, x_min - random.randint(0, 80))
                x_max = min(W, x_max + random.randint(0, 80))
                y_min = max(0, y_min - random.randint(0, 80))
                y_max = min(H, y_max + random.randint(0, 80))
                box = np.array([x_min, y_min, x_max, y_max])
            else:
                box = np.array([0, 0, 0,0])
        if self.transforms and not self.test:
            image, mask, box = self.transforms(image, mask, box)
        if self.transform:
            image = self.transform(image, mask)
        if self.target_transform and not self.test:
            mask = self.target_transform(mask)

        if self.test:
            return image
        else:
            return image, mask, box


def get_dataset(imgdir, maskdir=None, train=True, val=False, test=False,
                transforms=None, transform=None, target_transform=None):
    dataset = Gleason(imgdir=imgdir, maskdir=maskdir, train=train,
                      val=val, test=test, transforms=transforms,
                      transform=transform, target_transform=target_transform)

    return dataset


def get_transform(train):
    base_size = 1000
    crop_size = 768

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.ColorJitter(0.5, 0.5, 0.5, 0.5))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)