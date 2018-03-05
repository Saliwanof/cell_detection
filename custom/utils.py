import torch
import random
import numpy as np
from torchvision.transforms import ToPILImage

def conditioned_inverse(im):
    im = np.array(im)
    not_brightfield = np.mean(im) < 0.45
    if not_brightfield: im = 1 - im
    # im = torch.from_numpy(np.expand_dims(im, axis=0))
    imin, imax = np.amin(im), np.amax(im)
    k = 1. / (imax - imin)
    b = imin / (imin - imax)
    im = k * im + b
    
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
    
def get_weight_c(mask, factor01=(.5, .5)):
    total_count = mask.size
    class1_count = np.count_nonzero(mask)
    class0_count = total_count - class1_count
    class1_weight = total_count * factor[1] / class1_count
    class0_weight = total_count * factor[0] / class0_count
    
    weight_c = np.where(mask, class1_weight, class0_weight)
    
    return weight_c