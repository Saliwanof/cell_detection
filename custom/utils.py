import torch
import random
import numpy as np
from torchvision.transforms import ToPILImage

def conditioned_inverse(im):
    im = ToPILImage()(im)
    im = np.array(im)
    is_brightfield = np.mean(im) < 0.45
    if is_brightfield: im = 1 - im
    # im = torch.from_numpy(np.expand_dims(im, axis=0))
    
    return im

def random_crop(im, output_size=256, pad_value=0):
    h, w = im.shape
    th, tw = 256, 256
    if h < th:
        im = np.pad(im, ((0, th-h), (0, 0)), 'constant', constant_values=(0, pad_value))
    if w < tw:
        im = np.pad(im, ((0, 0), (0, tw-w)), 'constant', constant_values=(0, pad_value))
    h, w = im.shape
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    im = im[i:i+th, j:j+tw]
    
    return im.astype(float)