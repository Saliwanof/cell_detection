import torch
import random
import numpy as np
from torchvision.transforms import ToPILImage
from scipy.ndimage.interpolation import rotate

def to_brightfield(im):
    im = np.array(im)
    not_brightfield = np.mean(im) < 0.45
    if not_brightfield: im = 1 - im
    # im = torch.from_numpy(np.expand_dims(im, axis=0))
    imin, imax = np.amin(im), np.amax(im)
    k = 1. / (imax - imin)
    b = imin / (imin - imax)
    im = k * im + b
    
    return im

def random_crop_256(*argv, pad_values=None):
    th, tw = 256, 256
    ims = []
    for idx, im in enumerate(argv):
        h, w = im.shape
        pad_value = pad_values[idx]
        if h < th:
            im = np.pad(im, ((0, th-h), (0, 0)), 'constant', constant_values=(0, pad_value))
        if w < tw:
            im = np.pad(im, ((0, 0), (0, tw-w)), 'constant', constant_values=(0, pad_value))
        h, w = im.shape
        if idx==0:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        im = im[i:i+th, j:j+tw]
        ims.append(im)
    
    return ims

def random_rotate(*argv, pad_values=None):
    ims = []
    for idx, im in enumerate(argv):
        pad_value = pad_values[idx]
        if idx==0:
            angle = random.uniform(0., 90.)
        im = rotate(im, angle, reshape=False, cval=pad_value)
        ims.append(im)
    
    return ims

def random_flip(*argv):
    ud, lr = np.random.uniform(size=2) < .5
    ims = []
    for im in argv:
        if ud:
            im = np.flipud(im)
        if lr:
            im = np.fliplr(im)
        ims.append(im)
    
    return ims
    
def get_weight_c(mask, factor01=(.5, .5)):
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy()
    total_count = mask.size
    class1_count = np.count_nonzero(mask)
    class0_count = total_count - class1_count
    class1_weight = total_count * factor01[1] / class1_count
    class0_weight = total_count * factor01[0] / class0_count
    
    weight_c = np.where(mask, class1_weight, class0_weight)
    
    return weight_c