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