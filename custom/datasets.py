import torch
import h5py
import numpy as np
from os.path import isdir
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage.morphology import distance_transform_edt, binary_dilation, generate_binary_structure
from .utils import to_brightfield, random_crop_256, random_flip, random_rotate, random_zoom


class nuclei_dataset(torch.utils.data.Dataset):
    def __init__(self, dat, mode='train'):
        super(nuclei_dataset, self).__init__()
        self.dat = dat
        self.mode = mode # train, eval, full, test
        
    def __getitem__(self, n):
        if self.mode is 'eval':
            n = self.__len__() - 1 -n
            
        if self.mode is 'test':
            input = self.dat[n]
            input = np.expand_dims(input, axis=0)
            return input
        
        input, target, dist, wt = self.dat[n]
        input = to_brightfield(input)
        # input, target, dist = random_zoom(input, target, dist, pad_values=[1, 0, 257])
        input, target, dist = random_crop_256(input, target, dist, pad_values=[1, 0, 257])
        input, target, dist = random_flip(input, target, dist)
        input, target, dist = random_rotate(input, target, dist, pad_values=[1, 0, 257])
        input = np.expand_dims(input, axis=0)
        target = np.expand_dims(target, axis=0)
        dist = np.expand_dims(dist, axis=0)
        
        return input, target, dist, wt
        
    def __len__(self):
        n = len(self.dat)
        if self.mode is 'test' or self.mode is 'full': return n
        if self.mode is 'train': n = n * 0.8
        if self.mode is 'eval': n = n * 0.2
        
        return int(n)
        
class nuclei_data(object):
    def __init__(self, path, label=None, test=False):
        # path is like '../data/stage1_train/'
        gen_file = not list(Path(path).glob('dat.h5'))
        if gen_file:
            h5f_gen(path, test)
        path = path + 'dat.h5'
        self.h5f = h5py.File(path, 'r')
        self.im_ids = list(self.h5f['im_ids'])
        self.test = test
        
        if not test:
            self.im_labels = np.array(self.h5f['im_labels'])
            self.label_weights = get_label_weights(self.im_labels)
            if label is not None:
                idx = np.where(self.im_labels==label)[0]
                self.im_ids = [self.im_ids[i] for i in idx]
                self.label_weights = [self.label_weights[i] for i in idx]
    
    def __getitem__(self, n):
        im_id = self.im_ids[n]
        im = np.array(self.h5f[im_id+'.im'])
        if self.test:
            return im
        
        mask = np.array(self.h5f[im_id+'.mask'])
        dist = np.array(self.h5f[im_id+'.dist'])
        weight = self.label_weights[n]
        
        return im, mask, dist, weight
        
    def __len__(self):
        return len(self.im_ids)
    
    def close_h5f(self):
        self.h5f.close()
            
def h5f_gen(path, test=False):
    im_paths = Path(path).glob('*/images/*.png')                
    h5f = h5py.File(path+'dat.h5', 'w')
    im_ids = []
    for im_path in im_paths:
        im_id = im_path.parts[-3]
        im_ids.append(im_id)
        im_path = str(im_path)

        im = Image.open(im_path).convert('L')
        im = np.array(im) / 255.
        h5f.create_dataset(im_id+'.im', data=im)
        
        if not test:
            mask_paths = map(str, list(Path(path+im_id+'/masks/').glob('*.png')))
            mask, dist = get_mask_and_dist(mask_paths)
            h5f.create_dataset(im_id+'.mask', data=mask)
            h5f.create_dataset(im_id+'.dist', data=dist)
    im_ids = np.array(im_ids, dtype='object')
    dt = h5py.special_dtype(vlen=str)
    h5f.create_dataset('im_ids', data=im_ids, dtype=dt)
    if not test:
        label_file = h5py.File(path+'labels.h5', 'r')
        im_labels = label_file['im_label']
        h5f.create_dataset('im_labels', data=im_labels)
        label_file.close()

    h5f.close()    

def get_mask_and_dist(mask_paths):
    mask_paths = list(mask_paths)

    sample = np.array(Image.open(mask_paths[0]))
    mask = np.zeros_like(sample)
    dist = np.full(sample.shape, np.inf)

    for mask_path in mask_paths:
        mask_ = np.array(Image.open(mask_path))
        mask_ = np.where(mask_, 1, 0)

        dist_ = distance_transform_edt(mask_) + distance_transform_edt(1 - mask_)
        dist = np.dstack((dist, dist_))
        dist = np.amin(dist, axis=2)
        
        mask_d = binary_dilation(mask_, structure=generate_binary_structure(2, 2), iterations=2).astype(mask_.dtype)
        mask_ = mask_d - mask_
        mask = np.logical_or(mask_, mask)
        mask = np.where(mask, 1, 0)

    return mask, dist

def get_label_weights(labels):
    nl = labels.max() + 1
    total_count = labels.size
    weights = np.empty_like(labels, dtype=np.float_)
    for l in range(nl):
        l_count = np.count_nonzero(labels==l)
        weights[labels==l] = total_count * 1. / nl / l_count

    return weights

#










