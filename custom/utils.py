import torch
import random
import numpy as np
from torchvision.transforms import ToPILImage
from scipy.ndimage import rotate, zoom
from scipy import ndimage

def to_brightfield(im):
    im = np.array(im)
    not_brightfield = np.mean(im) < 0.42
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

def random_zoom(*argv, pad_values=None):
    ims = []
    scales = np.random.uniform(.8, 1.5, 2)
    for idx, im in enumerate(argv):
        pad_value = pad_values[idx]
        im = zoom(im, scales, order=3, cval=pad_value)
        ims.append(im)
    
    return ims

def get_weight_c(mask, factor01=(.5, .5)):
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy()
    total_count = mask.size
    class1_count = np.count_nonzero(mask)
    class0_count = total_count - class1_count
    try:
        class1_weight = total_count * factor01[1] / class1_count
        class0_weight = total_count * factor01[0] / class0_count
    except:
        class1_weight = 100
        class0_weight = 100
    
    weight_c = np.where(mask, class1_weight, class0_weight)
    
    return weight_c

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
        
    return ' '.join([str(i) for i in run_lengths])

def mask_seperation(mask):
    mask = 1 - mask
    labels, nlabels = ndimage.label(mask)
    label_arrays = []
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)
    
    return label_arrays