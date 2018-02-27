import numpy as np
from torchvision.transforms import ToPILImage, ToTensor

def conditioned_inverse(im):
    im = ToPILImage()(im)
    im = np.array(im)
    is_brightfield = np.mean(im) < 0.45
    if is_brightfield: im = 1 - im
    im = ToTensor()(im)
    
    return im