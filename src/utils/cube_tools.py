from glob import glob
import nibabel as nib
import numpy as np
import pandas as pd
import os
from random import shuffle
import datetime

from utils.Normalization import normalization
from utils.Resample import resample, mask2binary
from utils.WhiteStripeNormalization import ws_normalize
from utils.cube_tools import get_center, img2cube

from concurrent.futures import ProcessPoolExecutor


def process_patient(i, patient, file_list, norm, do_resample, features, target_voxel, brain_mask):
    ADC_path, FLAIR_path, b1000_path, BRAIN_path, INFARCT_path = sorted(
            [path for path in file_list if patient in path])

    try:
        # Data Preparing
        ADC_array = nib.load(ADC_path).get_data()
        FLAIR_array = nib.load(FLAIR_path).get_data()
        BRAIN_array = nib.load(BRAIN_path).get_data()

        INFARCT_nii = nib.load(INFARCT_path)
        INFARCT_array = INFARCT_nii.get_data()

        origin_voxel_size = INFARCT_nii.header.get_zooms()

        # Pre-processing (1)- Resampling Voxel Size
        if do_resample:
            ADC_array = resample(ADC_array, origin_voxel_size, target_voxel)
            FLAIR_array = resample(FLAIR_array, origin_voxel_size, target_voxel)
            BRAIN_array = resample(BRAIN_array, origin_voxel_size, target_voxel)
            INFARCT_array = resample(INFARCT_array, origin_voxel_size, target_voxel)

            if i % 20 == 0:
                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : Voxel Size Resampling ({})".format(time))

        # Make Mask to Binary (0 or 1)
        BRAIN_array = mask2binary(BRAIN_array)
        INFARCT_array = mask2binary(INFARCT_array)
        print(">>> INFARCT mask shape :", INFARCT_array.shape)

        if norm == 'ws':
            FLAIR_array = ws_normalize(FLAIR_array, 'FLAIR', BRAIN_array)
            
            if i % 20 == 0:
                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : White-stripe Normalization ({})".format(time))
            
        elif norm == 'new':
            FLAIR_array = normalization(FLAIR_array, INFARCT_array)

            if i % 20 == 0:
                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : New Normalization ({})".format(time))
            
        else:
            raise ValueError("Value of 'norm' parameter should be 'new' of 'ws'")

        if brain_mask:
            center = get_center(mask_array=BRAIN_array)
        else:
            center = get_center(mask_array=INFARCT_array)
        ADC_cube_array = img2cube(img_array=ADC_array, cube_center=center, cube_shape=(64, 64, 10))
        FLAIR_cube_array = img2cube(img_array=FLAIR_array, cube_center=center, cube_shape=(64, 64, 10))
        return (ADC_cube_array,FLAIR_cube_array, i, patient)

    except Exception as ex:
        ex.args = (*[a for a in ex.args], i, patient)
        return ex

def process_patient_wrapper(X_ADC, X_FLAIR, y, patient_num, error_patient, patient_list, file_list, patient_type,
                            norm, do_resample, features, target_voxel, brain_mask):
    assert patient_type in ["Positive", "Negative", "Test"]
    target = 1 if patient_type == "Positive" else 0

    futures = []
    with ProcessPoolExecutor() as executor:
        for i, patient in enumerate(patient_list):
            time = str(datetime.datetime.now()).split()[1].split('.')[0]
            print("Processing [{}/{}] Image of {} Patient... ({})".format(i + 1, len(patient_list), patient_type, time))

            futures.append(executor.submit(process_patient, i, patient, file_list, norm, do_resample, features, target_voxel, brain_mask))

    output = [future.result() for future in futures]
    for out in output:
        time = str(datetime.datetime.now()).split()[1].split('.')[0]
        if isinstance(out, Exception):
            msg, i, patient = out.args
            print("Error !!! [Patient Number : {}] ({})".format(i + 1, time))
            error_patient.append(patient)
            print(msg)
        else:
            ADC_cube_array, FLAIR_cube_array, i, patient = out
            X_ADC.append(ADC_cube_array)
            X_FLAIR.append(FLAIR_cube_array)
            if patient_type in ["Positive", "Negative"]:
                y.append(target)
            patient_num.append(patient)


# Feature Extraction for Train
def train_cube_loader(pos_dir='/data/train/positive/', neg_dir='/data/train/negative/', norm='new',
                      do_resample=True, do_shuffle=True,
                      features = ['firstorder', 'shape'], target_voxel = (0.65, 0.65, 3), brain_mask=True):
    # File List
    pos_file_list = glob(pos_dir + "*")
    neg_file_list = glob(neg_dir + "*")

    pos_patient_list = sorted(list(set([path.split('_')[0] for path in os.listdir(pos_dir)])))
    neg_patient_list = sorted(list(set([path.split('_')[0] for path in os.listdir(neg_dir)])))

    # Data Container
    X_ADC = []
    X_FLAIR = []
    y = []
    patient_num = []
    error_patient = []

    process_patient_wrapper(X_ADC, X_FLAIR, y, patient_num, error_patient, pos_patient_list, pos_file_list, "Positive",
                            norm, do_resample, features, target_voxel, brain_mask)

    process_patient_wrapper(X_ADC, X_FLAIR, y, patient_num, error_patient, neg_patient_list, neg_file_list, "Negative",
                            norm, do_resample, features, target_voxel, brain_mask)

    if do_shuffle:
        shuffle_list = [[X1_value, X2_value, y_value, num] for X1_value, X2_value, y_value, num in zip(X_ADC,X_FLAIR, y, patient_num)]
        shuffle(shuffle_list)

        X_ADC = [value[0] for value in shuffle_list]
        X_FLAIR = [value[1] for value in shuffle_list]
        y = [value[2] for value in shuffle_list]
        patient_num = [value[3] for value in shuffle_list]

    X_ADC = np.array(X_ADC)
    X_FLAIR = np.array(X_FLAIR)
    y = np.array(y)
    time = str(datetime.datetime.now()).split()[1].split('.')[0]
    print("Created X of shape {}, {}  and y of shape {} ({})".format(X_ADC.shape, X_FLAIR, y.shape, time))

    return X_ADC, X_FLAIR, y



# Feature Extraction for Inference
def test_cube_loader(test_dir='/data/test/', norm='new',
                     do_resample=True, do_shuffle=False,
                     features = ['firstorder', 'shape'], target_voxel = (0.65, 0.65, 3), brain_mask=True):
    # File List
    test_file_list = glob(test_dir + "*")
    test_patient_list = list(set([path.split('_')[0] for path in os.listdir(test_dir)]))

    # Data Container
    X_ADC = []
    X_FLAIR = []
    patient_num = []
    error_patient = []

    process_patient_wrapper(X_ADC, X_FLAIR, [], patient_num, error_patient, test_patient_list, test_file_list, "Test",
                            norm, do_resample, features, target_voxel, brain_mask)

    X_ADC = np.array(X_ADC)
    X_FLAIR = np.array(X_FLAIR)
    
    return X_ADC, X_FLAIR, patient_num, error_patient

def get_center(mask_array):
    x_values, y_values, z_values = np.where(mask_array > 0.5)
    
    # Mask has 0 or 1
    if len(x_values) > 0:
        x_center = int(np.mean(np.array(x_values)))
        y_center = int(np.mean(np.array(y_values)))
        z_center = int(np.mean(np.array(z_values)))

        return [x_center, y_center, z_center]
    
    # Mask only has 0
    else:
        return None


def cube_indexing(center_point, cube_size):
    half_size = cube_size // 2

    # When Cube Size is Even Number
    if (half_size * 2) == cube_size:
        indexing = [center_point - half_size, center_point + half_size]
    
    # When Cube Size is Odd Number
    else: 
        indexing = [center_point - half_size, center_point + half_size + 1]
    
    return indexing

    
def img2cube(img_array, cube_center, cube_shape):
    cube_array = np.copy(img_array)

    x_size, y_size, z_size = np.array(cube_shape)

    if not (x_size > img_array.shape[0] or y_size > img_array.shape[1] or z_size > img_array.shape[2]):
        x_center, y_center, z_center = cube_center
        
        x_index = cube_indexing(x_center, x_size)
        y_index = cube_indexing(y_center, y_size)
        z_index = cube_indexing(z_center, z_size)

        cube_array = cube_array[x_index[0] : x_index[1],
                                y_index[0] : y_index[1],
                                z_index[0] : z_index[1]]
        
        return cube_array
    
    else:
        print("Please reset the shape of cube (Cube is out of image)")
        return None
