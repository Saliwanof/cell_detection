import pathlib
import imageio
import scipy
import numpy as np
import pandas as pd
from imageio import imread
from scipy.ndimage.morphology import distance_transform_edt as dte 

def nuclei_dataset(main_path='./data/stage1_train/'):
    df = get_im_paths(main_path)
    df['mask'] = None
    df['weight'] = None
    df['im'] = None
    for idx in range(len(df)):
        im_id = df.loc[idx]['im_id']
        im_path = df.loc[idx]['im_path']
        mask_paths = list(pathlib.Path(main_path + im_id + '/masks/').glob('*.png'))
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
    mask = []
    dist = []
    for mask_path in mask_paths:
        mask_ = imread(mask_path)
        mask_ = np.where(mask_, 1, 0)
        mask = np.concatenate(mask, mask_, axis=2)
        mask = np.amax(mask, axis=2)
        dist_ = dte(mask_) + dte(1 - mask_)
        dist = np.concatenate(dist, dist_, axis=2)
        dist = np.amin(dist, axis=2)
    weight = get_weight_c(mask) + 10 * np.exp(-2 * np.power(dist, 2) / np.power(sigma, 2))
    
    return mask, weight

def get_weight_c(mask):
    total_count = mask.size
    class1_count = np.count_nonzero(mask)
    class0_count = total_count - class1_count
    class1_weight = total_count * .5 / class1_count
    class0_weight = total_count * .5 / class0_count
    weight_c = np.where(mask, class1_weight, class0_weight)
    
    return weight_c







    