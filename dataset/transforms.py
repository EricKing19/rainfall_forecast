import torch
import numpy as np

import numbers
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img


class ToTensor(object):
    def __call__(self, *img):
        img = list(img)
        for i in range(len(img)):
            if isinstance(img[i], np.ndarray):
                img[i] = torch.from_numpy(img[i])
            else:
                raise TypeError('Data should be ndarray.')
        return tuple(img)


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *img):
        th, tw = self.size
        h, w = img[0].shape[-2], img[0].shape[-1]
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = list(img)
        for i in range(len(img)):
            img[i] = self.crop(img[i], y1, x1, y1+ th, x1 + tw)
        return tuple(img)

    def crop(self, im, x_start, y_start, x_end, y_end):
        if len(im.shape) == 3:
            return im[:, x_start:x_end, y_start:y_end]
        else:
            return im[x_start:x_end, y_start:y_end]

class Normalize(object):
    def __init__(self):
        self.mean = [250., 50., 0., 0.]
        self.std = [60., 50., 50., 50.]

    def __call__(self, *img):
        img = list(img)
        for i in range(4):
            img[i] = (img[i] - self.mean[i]) / self.std[i]
            img[i + 4] = (img[i + 4] - self.mean[i]) /self.std[i]
        return tuple(img)

if __name__ == '__main__':
    a = np.arange(2*10*10)
    a.resize([2, 10, 10])
    a = a.astype(np.float32)
    b = a + 1

    t = Compose([
        RandomCrop(5),
        ToTensor(),
    ])
    c, d = t(*[a, b])
