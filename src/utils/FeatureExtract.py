from __future__ import print_function
import radiomics, logging
from radiomics import featureextractor
import radiomics
import SimpleITK as sitk
from radiomics import firstorder, glcm, shape, glrlm, glszm, ngtdm, gldm
import numpy as np

radiomics.logging.setLevel(logging.ERROR)

def Numpy2Itk(array):
    '''
    :param array: numpy array format
    :return: simple itk image type format
    '''
    return sitk.GetImageFromArray(array)

def feature_extract(image_origin, image_mask, features = ['firstorder', 'glcm', 'glszm', 'glrlm', 'ngtdm', 'shape']):
    '''
    :param image_origin: image_array (numpy array)
    :param image_mask: mask_array (numpy array)
    :subject: subject name
    :return: whole features, featureVector, make csv_file
    '''
    image = Numpy2Itk(image_origin)
    mask = Numpy2Itk(image_mask)
    
    settings = {}
    settings['binwidth'] = 25
    settings['resampledPixelSpacing'] = None
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True
    
    extractor = featureextractor.RadiomicsFeaturesExtractor(**settings)
    extractor.settings['enableCExtensions'] = True
    
    for feature in features:
        extractor.enableFeatureClassByName(feature.lower())
        
    featureVector = extractor.execute(image, mask)
    
    cols = []; feats = []
    for feature in features:
        for featureName in featureVector.keys():
            if feature in featureName:
                cols.append(featureName)
                feats.append(featureVector[featureName])
    return feats, cols
