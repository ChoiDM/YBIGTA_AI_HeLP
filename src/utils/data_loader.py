from glob import glob
import nibabel as nib
import numpy as np
import pandas as pd
import os
from random import shuffle
import random
import datetime

from utils.Normalization import normalization, min_max
from utils.Resample import resample, mask2binary
from utils.WhiteStripeNormalization import ws_normalize
from utils.FeatureExtract import feature_extract
from utils.cube_tools import get_center, img2cube

from concurrent.futures import ProcessPoolExecutor


def process_patient(i, patient, file_list, norm, do_resample, features, target_voxel):
    ADC_path, FLAIR_path, b1000_path, BRAIN_path, INFARCT_path = \
        sorted([path for path in file_list if patient in path])

    try:
        # Data Preparing
        ADC_array = nib.load(ADC_path).get_data()
        FLAIR_array = nib.load(FLAIR_path).get_data()
        BRAIN_array = nib.load(BRAIN_path).get_data()

        INFARCT_nii = nib.load(INFARCT_path)
        INFARCT_array = INFARCT_nii.get_data()

        origin_voxel_size = INFARCT_nii.header.get_zooms()


        # Pre-processing (1) - Resampling Voxel Size
        if do_resample:
            ADC_array = resample(ADC_array, origin_voxel_size, target_voxel)
            FLAIR_array = resample(FLAIR_array, origin_voxel_size, target_voxel)
            BRAIN_array = resample(BRAIN_array, origin_voxel_size, target_voxel)
            INFARCT_array = resample(INFARCT_array, origin_voxel_size, target_voxel)


        # Make Mask to Binary (0 or 1)
        BRAIN_array = mask2binary(BRAIN_array)
        INFARCT_array = mask2binary(INFARCT_array)


        # Pre-processing (2) - Normalization
        if norm == 'ws':
            FLAIR_array = ws_normalize(FLAIR_array, 'FLAIR', BRAIN_array)
        elif norm == 'new':
            FLAIR_array = normalization(FLAIR_array, INFARCT_array, size = 8)
        else:
            raise ValueError("Value of 'norm' parameter should be 'new' of 'ws'")


        # Feature Extraction by Radiomics
        ADC_values, ADC_columns = feature_extract(ADC_array, INFARCT_array, features)
        FLAIR_values, FLAIR_columns = feature_extract(FLAIR_array, INFARCT_array, features)

        total_values = ADC_values + FLAIR_values
        # total_columns = ['ADC_' + col for col in ADC_columns] + ['FLAIR_' + col for col in ADC_columns]

        return (total_values, i, patient)

    except Exception as ex:
        ex.args = (*[a for a in ex.args], i, patient)
        return ex


def process_patient_wrapper(X, y, patient_num, error_patient, patient_list, file_list, patient_type,
                            norm, do_resample, features, target_voxel):

    assert patient_type in ["Positive", "Negative", "Test"]
    target = 1 if patient_type == "Positive" else 0

    futures = []
    with ProcessPoolExecutor() as executor:
        for i, patient in enumerate(patient_list):
            futures.append(executor.submit(process_patient, i, patient, file_list, norm, do_resample, features, target_voxel))

    output = [future.result() for future in futures]
    for out in output:

        time = str(datetime.datetime.now()).split()[1].split('.')[0]
        if isinstance(out, Exception):
            msg, i, patient = out.args
            print("!!! Error : [Patient Number : {}] ({})".format(i + 1, time))
            error_patient.append(patient)
            print(msg)

        else:
            total_values, i, patient = out
            X.append(total_values)
            if patient_type in ["Positive", "Negative"]:
                y.append(target)
            patient_num.append(patient)


# Feature Extraction for Train
def train_data_loader(pos_dir='/data/train/positive/', neg_dir='/data/train/negative/', norm='new',
                      do_resample=True, do_shuffle=True, do_minmax=True,
                      features = ['firstorder', 'shape'], target_voxel = (0.65, 0.65, 3), path="/data"):
    # File List
    pos_file_list = glob(pos_dir + "*")
    neg_file_list = glob(neg_dir + "*")

    pos_patient_list = sorted(list(set([path.split('_')[0] for path in os.listdir(pos_dir)])))
    neg_patient_list = sorted(list(set([path.split('_')[0] for path in os.listdir(neg_dir)])))

    # Data Container
    X = []
    y = []
    patient_num = []
    error_patient = []

    process_patient_wrapper(X, y, patient_num, error_patient, pos_patient_list, pos_file_list, "Positive",
                            norm, do_resample, features, target_voxel)

    process_patient_wrapper(X, y, patient_num, error_patient, neg_patient_list, neg_file_list, "Negative",
                            norm, do_resample, features, target_voxel)

    if do_shuffle:
        shuffle_list = [[X_value, y_value, num] for X_value, y_value, num in zip(X, y, patient_num)]
        shuffle(shuffle_list)

        X = [value[0] for value in shuffle_list]
        y = [value[1] for value in shuffle_list]
        patient_num = [value[2] for value in shuffle_list]

    X = np.array(X)
    y = np.array(y)
    time = str(datetime.datetime.now()).split()[1].split('.')[0]
    print("Created X of shape {} and y of shape {} ({})".format(X.shape, y.shape, time))
    
    if do_minmax:
        X = min_max(X, mode = 'train', path=path)
        
    return X, y



# Feature Extraction for Inference
def test_data_loader(test_dir='/data/test/', norm='new',
                     do_resample=True, do_minmax=True,
                     features = ['firstorder', 'shape'], target_voxel = (0.65, 0.65, 3), path="/data"):
    
    # File List
    test_file_list = glob(test_dir + "*")
    test_patient_list = list(set([path.split('_')[0] for path in os.listdir(test_dir)]))

    # Data Container
    X = []
    patient_num = []
    error_patient = []

    process_patient_wrapper(X, [], patient_num, error_patient, test_patient_list, test_file_list, "Test",
                            norm, do_resample, features, target_voxel)

    X = np.array(X)
    
    if do_minmax:
        X = min_max(X, mode = 'test', path=path)
    
    return X, patient_num, error_patient


# ------------------------------------------ cnn -------------------------------------------------

def get_cube(patient_num, data_dir, cube_shape, norm, mode, target_voxel=(0.65, 0.65, 3)):
    ADC_path, FLAIR_path, b1000_path, BRAIN_path, INFARCT_path = \
        sorted([path for path in data_dir if patient_num in path])

    if mode == 'train':
        if 'positive' in ADC_path:
            label = 1
        
        elif 'negative' in ADC_path:
            label = 0

    img_array1 = nib.load(ADC_path).get_data()
    img_array2 = nib.load(FLAIR_path).get_data()

    mask_nii = nib.load(INFARCT_path)
    origin_voxel_size = mask_nii.header.get_zooms()
    mask_array = mask_nii.get_data()

    img_array1 = resample(img_array1, origin_voxel_size, target_voxel)
    img_array2 = resample(img_array2, origin_voxel_size, target_voxel)
    mask_array = resample(mask_array, origin_voxel_size, target_voxel)

    mask_array = mask2binary(mask_array)

    if norm == 'ws':
        BRAIN_array = nib.load(BRAIN_path).get_data()
        BRAIN_array = resample(BRAIN_array, origin_voxel_size, target_voxel)
        img_array2 = ws_normalize(img_array2, 'FLAIR', BRAIN_array)

    elif norm == 'new':
        img_array2 = normalization(img_array2, mask_array, size= 8)

    cube_center = get_center(mask_array)
    cube_array1 = img2cube(img_array1, cube_center, cube_shape)
    cube_array2 = img2cube(img_array2, cube_center, cube_shape)
    
    cube_array1 = cube_array1[..., None]
    cube_array2 = cube_array2[..., None]
    cube_array = np.concatenate([cube_array1, cube_array2], axis=3)

    if mode == 'train':
        return cube_array, label
    
    elif mode == 'test':
        return cube_array
    
    else:
        raise ValueError("value of parameter 'mode' must be 'train' or 'test'")


def data_generator(batch_size, mode, data_dir, cube_shape, norm, target_voxel = (0.65, 0.65, 3), seq = None):

    patient_list = []
    for path in data_dir:
        ID = path.split(os.sep)[-1].split('_')[0]
        
        if ID not in patient_list:
            patient_list.append(ID)

    if mode == 'train':
        shuffle(patient_list)

        while True:
            batch_imgs = []
            batch_labels = []
            batch_idx = np.random.choice(len(patient_list), batch_size)
            
            for index in batch_idx:
                cube_array, label = get_cube(patient_list[index], data_dir, cube_shape, norm, mode, target_voxel)
                
                if cube_array.shape != (32,32,16,2) :
                    print("Error!!! with {} patient : Shape is".format(index, cube_array.shape))
                    continue
                
                batch_imgs.append(cube_array)
                batch_labels.append(label)
           
            batch_imgs = np.array(batch_imgs)
            batch_labels = np.array(batch_labels)

            if seq is not None:
                batch_imgs = seq.augment_images(batch_imgs)

            yield batch_imgs, batch_labels
            
    if mode == 'test':
        
        batch_imgs = []

        for index in range(len(patient_list)):
            cube_array = get_cube(patient_list[index], data_dir, cube_shape, norm, mode, target_voxel)
                
            if cube_array.shape != (32,32,16,2) :
                print("\nError!!! with {} patient : Shape is {}".format(index, cube_array.shape))
                continue
                    
            batch_imgs.append(cube_array)
            
        return np.array(batch_imgs)
