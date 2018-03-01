import pathlib
import imageio
import scipy
import torch
import warnings
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from imageio import imread
from .utils import conditioned_inverse, random_crop
from scipy.ndimage.morphology import distance_transform_edt as dte 

class nuclei_dataset(torch.utils.data.Dataset):
    def __init__(self, nuclei_dataset_df, train=True, from_saved=False):
        super(nuclei_dataset, self).__init__()
        self.df = nuclei_dataset_df
        # self.df.to_csv(path + 'data.csv')
        self.train = train
        self.transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            # transforms.RandomCrop(256),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip()
                            ])
    def __getitem__(self, n):
        if not self.train:
            n = self.__len__() - 1 -n
        input = np.array(self.df['im'][n])
        input = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Grayscale(),
                                    transforms.Lambda(lambda x: np.array(x) / 255.),
                                    transforms.Lambda(conditioned_inverse)
                                    ])(input) # np.ndarray
        target = np.array(self.df['mask'][n])
        weight = np.array(self.df['weight'][n])
        
        stack = np.stack((input, target, weight), axis=2)
        # if self.train: stack = self.transforms(stack)
        # stack = random_crop(stack)
        
        to_tensor = lambda x: torch.from_numpy(np.expand_dims(x, axis=0))
        input = to_tensor(random_crop(input)).float()
        target = to_tensor(random_crop(target)).float()
        weight = to_tensor(random_crop(weight, pad_value=1)).float()
        
        return input, target, weight
        
    def __len__(self):
        n = len(self.df.index)
        if self.train: n = n * 0.8
        else: n = n * 0.2
        
        return int(n)
#

def nuclei_dataset_gen(path='../data/stage1_train/'):
    df = get_im_paths(path)
    df['mask'] = None
    df['weight'] = None
    df['im'] = None
    for idx in range(len(df)):
        im_id = df.loc[idx]['im_id']
        im_path = df.loc[idx]['im_path']
        mask_paths = list(pathlib.Path(path + im_id + '/masks/').glob('*.png'))
        mask_paths = [str(x) for x in mask_paths]
        mask, weight = make_mask_and_wt(mask_paths)
        df.loc[idx]['mask'] = mask
        df.loc[idx]['weight'] = weight
        df.loc[idx]['im'] = np.array(imread(im_path))
    
    return df

#
def get_im_paths(main_path):
    im_paths = list(pathlib.Path(main_path).glob('*/images/*.png'))
    im_ids = [x.parts[-3] for x in im_paths]
    im_paths = [str(x) for x in im_paths]
    df = pd.DataFrame({'im_id':im_ids, 'im_path':im_paths})
    
    return df

def make_mask_and_wt(mask_paths, sigma=5):
    sample = imread(mask_paths[0])
    
    mask = np.zeros_like(sample)
    dist = np.full_like(sample, np.inf)
    
    for mask_path in mask_paths:
        
        mask_ = np.array(imread(mask_path))
        # mask_[np.isnan(mask_)] = 0
        mask_ = np.where(mask_, 1, 0)
        mask = np.logical_or(mask_, mask)
        mask = np.where(mask, 1, 0)
        
        dist_ = dte(mask_) + dte(1 - mask_)
        dist = np.dstack((dist, dist_))
        dist = np.amin(dist, axis=2)
    
    dist_large_value_idx = dist > 10
    dist[dist_large_value_idx] = 0
    weight = np.where(dist_large_value_idx, .003, 10 * np.exp(-2 * np.power(dist, 2) / np.power(sigma, 2)))
    weight = get_weight_c(mask) + np.where(np.isnan(weight), 1, weight)
    
    return mask, weight

def get_weight_c(mask):
    total_count = mask.size
    class1_count = np.count_nonzero(mask)
    class0_count = total_count - class1_count
    class1_weight = total_count * 5 / class1_count
    try:
        class0_weight = total_count * 5 / class0_count
    except:
        warnings.warn("class_0 count zero!")
        class0_weight = 0
    weight_c = np.where(mask, class1_weight, class0_weight)
    
    return weight_c







    