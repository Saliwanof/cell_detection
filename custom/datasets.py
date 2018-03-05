import h5py
import numpy as np
from os.path import isdir
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage.morphology import distance_transform_edt as dte 
from .utils import conditioned_inverse, random_crop


class nuclei_dataset(torch.utils.data.Dataset):
    def __init__(self, dat, train=True):
        super(nuclei_dataset, self).__init__()
        self.dat = dat
        self.train = train
        # self.transforms = transforms.Compose([
                            # transforms.ToPILImage(),
                            # # transforms.RandomCrop(256),
                            # transforms.RandomVerticalFlip(),
                            # transforms.RandomHorizontalFlip()
                            # ])
    def __getitem__(self, n):
        if not self.train:
            n = self.__len__() - 1 -n
        input, target, dist = self.dat[n]
        # transforms on input needed
        # transforms for augmentation needed
        
        return input, target, dist
        
    def __len__(self):
        n = len(self.dat)
        if self.train: n = n * 0.8
        else: n = n * 0.2
        
        return int(n)
        
class nuclei_data(object):
    def __init__(self, path):
        # path is like '../data/stage1_train/[dat.h5]'
        gen_file = not list(Path(path).glob('dat.h5'))
        if gen_file:
            h5f_gen(path)
            path = path + 'dat.h5'
        self.h5f = h5py.File(path, 'r')
        self.im_ids = list(self.h5f['im_ids'])
    #
    def __getitem__(self, n):
        im_id = self.im_ids[n]
        im = np.array(self.h5f[im_id+'.im'])
        mask = np.array(self.h5f[im_id+'.mask'])
        dist = np.array(self.h5f[im_id+'.dist'])
        
        return im, mask, dist 
        
    def __len__(self):
        return len(self.im_ids)
    
    def h5f_gen(path):
        im_paths = Path(path).glob('*/images/*.png')                
        h5f = h5py.File(path+'dat.h5', 'w')
        im_ids = []
        for im_path in im_paths:
            im_id = im_path.parts[-3]
            im_ids.append(im_id)
            im_path = str(im_path)
            
            im = Image.open(im_path).convert('L')
            h5f.create_dataset(im_id+'.im', data=np.array(im))
            
            mask_paths = Path(path+im_id).glob('/masks/*.png')
            mask, dist = get_mask_and_dist(mask_paths)
            h5f.create_dataset(im_id+'.mask', data=mask)
            h5f.create_dataset(im_id+'.dist', data=dist)
        h5f.create_dataset('im_ids', data=im_ids)
        
        h5f.close()    
        
                
    def get_mask_and_dist(mask_paths):
        mask_paths = list(mask_paths)
        
        sample = np.array(Image.open(mask_paths[0]))
        mask = np.zeros_like(sample)
        dist = np.full_like(sample, np.inf)
        
        for mask_path in mask_paths:
            mask_ = np.array(Image.open(mask_path))
            mask_ = np.where(mask_, 1, 0)
            mask = np.logical_or(mask_, mask)
            mask = np.where(mask, 1, 0)
        
            dist_ = dte(mask_) + dte(1 - mask_)
            dist = np.dstack((dist, dist_))
            dist = np.amin(dist, axis=2)
            
        return mask, dist
        


#










