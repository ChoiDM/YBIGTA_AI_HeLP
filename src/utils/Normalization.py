"""
intensity_normalization.normalize.whitestripe
Use the White Stripe method outlined in [1] to normalize
the intensity of an MR image
References:
﻿   [1] R. T. Shinohara, E. M. Sweeney, J. Goldsmith, N. Shiee,
        F. J. Mateen, P. A. Calabresi, S. Jarso, D. L. Pham,
        D. S. Reich, and C. M. Crainiceanu, “Statistical normalization
        techniques for magnetic resonance imaging,” NeuroImage Clin.,
        vol. 6, pp. 9–19, 2014.
Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Apr 27, 2018
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd


from utils import io
from utils import hist

class NormalizationError(Exception):
    pass


def ws_normalize(img_data, contrast, mask_data):
    '''
    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        contrast (str): contrast of img (e.g., T1)
        mask (nibabel.nifti1.Nifti1Image): brainmask for img (None is default, for skull-stripped img)
    '''

    indices = whitestripe(img_data, contrast, mask_data)
    normalized = whitestripe_norm(img_data, indices)

    # output the last normalized image (mostly for testing purposes)
    return normalized


def whitestripe(img_data, contrast, mask_data, width=0.05):
    """
    find the "(normal appearing) white (matter) stripe" of the input MR image
    and return the indices
    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        contrast (str): contrast of img (e.g., T1)
        mask (nibabel.nifti1.Nifti1Image): brainmask for img (None is default, for skull-stripped img)
        width (float): width quantile for the "white (matter) stripe"
        width_l (float): lower bound for width (default None, derives from width)
        width_u (float): upper bound for width (default None, derives from width)
    Returns:
        ws_ind (np.ndarray): the white stripe indices (boolean mask)
    """

    width_l = width
    width_u = width


    masked = img_data * mask_data
    voi = img_data[mask_data == np.unique(mask_data)[1]]

    if contrast.lower() in ['t1', 'flair', 'last']:
        mode = hist.get_last_mode(voi)

    elif contrast.lower() in ['t2', 'largest']:
        mode = hist.get_largest_mode(voi)

    elif contrast.lower() in ['md', 'first']:
        mode = hist.get_first_mode(voi)

    else:
        raise NormalizationError('Contrast {} not valid, needs to be T1, T2, FA, or MD')

    img_mode_q = np.mean(voi < mode)
    ws = np.percentile(voi, (max(img_mode_q - width_l, 0) * 100, min(img_mode_q + width_u, 1) * 100))
    ws_ind = np.logical_and(masked > ws[0], masked < ws[1])

    if len(ws_ind) == 0:
        raise NormalizationError('WhiteStripe failed to find any valid indices!')

    return ws_ind


def whitestripe_norm(img_data, indices):
    """
    use the whitestripe indices to standardize the data (i.e., subtract the
    mean of the values in the indices and divide by the std of those values)
    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        indices (np.ndarray): whitestripe indices (see whitestripe func)
    Returns:
        norm_img (nibabel.nifti1.Nifti1Image): normalized image in nifti format
    """

    mu = np.mean(img_data[indices])
    sig = np.std(img_data[indices])
    norm_img = (img_data - mu)/sig

    return norm_img

def change_values(expanded_mask, center_index, size):
    x_idx, y_idx = center_index

    for i in range(-size, size+1):
        for j in range(-size, size+1):
            expanded_mask[x_idx+i, y_idx+j] = 1


def lesion_based_normalization(img_array, mask_array, size):

    n_target = 0
    sum_values = 0

    for z_idx in range(mask_array.shape[-1]):

        mask_slice = mask_array[:, :, z_idx]
        expanded_mask = np.zeros_like(mask_slice)

        # If slice has infarct mask
        if len(np.unique(mask_slice)) > 1:
            
            for x_idx in range(mask_slice.shape[0]):
                for y_idx in range(mask_slice.shape[1]):

                    # If pixel value is 1.0 (infarct mask)
                    if (mask_slice[x_idx, y_idx] != 0.0) and (x_idx+size < mask_slice.shape[0]) and (y_idx+size < mask_slice.shape[1]):
                        change_values(expanded_mask, [x_idx, y_idx], size)
            
            expanded_mask[mask_slice != 0] = 0
            
            target_values = img_array[:, :, z_idx][mask_slice != 0]

            n_target += len(target_values)
            sum_values += sum(target_values)


    if n_target > 0:
        norm_factor = (sum_values // n_target)
        return img_array / norm_factor
    
    else:
        print(">>> Unable to do normalization : No value 1 in mask array.")
        return img_array

    
def min_max(X_array, mode, path = "/data"):
    df = pd.DataFrame(X_array)

    if mode == 'train':
        min_values = df.min()
        max_values = df.max()

        np.save(path + '/model/min.npy', np.array(min_values))
        np.save(path + '/model/max.npy', np.array(max_values))

        df_norm =  (df - min_values) / (max_values - min_values)
        df_norm = df_norm.dropna(axis=1)
    
    elif mode == 'test':
        min_values = np.load(path + '/model/min.npy')
        max_values = np.load(path + '/model/max.npy')
        
        df_norm =  (df - min_values) / (max_values - min_values)
        df_norm = df_norm.dropna(axis=1)
    
    else:
        raise ValueError("value of 'mode' parameter must be 'train' or 'test'")
    
    return df_norm.values
