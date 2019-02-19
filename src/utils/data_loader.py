from glob import glob
import nibabel as nib
import numpy as np
import pandas as pd
import os
from random import shuffle
import datetime

from utils.N4Correction import n4correction
from utils.Resample import resample, mask2binary
from utils.WhiteStripeNormalization import ws_normalize
from utils.FeatureExtract import feature_extract
from utils.Normalization import normalization


# Feature Extraction for Train
def train_data_loader(pos_dir='/data/train/positive/', neg_dir='/data/train/negative/', norm='new',
                      do_resample=True, do_shuffle=True,
                      features = ['firstorder', 'shape'], target_voxel = (0.65, 0.65, 3)):
    # File List
    pos_file_list = glob(pos_dir + "*")
    neg_file_list = glob(neg_dir + "*")

    pos_patient_list = sorted(list(set([path.split('_')[0] for path in os.listdir(pos_dir)])))
    neg_patient_list = sorted(list(set([path.split('_')[0] for path in os.listdir(neg_dir)])))

    # Data Container
    X = []
    y = []
    patient_num = []

    for i, pos_patient in enumerate(pos_patient_list):
        time = str(datetime.datetime.now()).split()[1].split('.')[0]
        print("Processing [{0}/{1}] Image of Positive Patient... ({2})".format(i + 1, len(pos_patient_list), time))

        ADC_path, FLAIR_path, b1000_path, BRAIN_path, INFARCT_path = sorted(
            [path for path in pos_file_list if pos_patient in path])

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

                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : Voxel Size Resampling ({})".format(time))

            # Make Mask to Binary (0 or 1)
            BRAIN_array = mask2binary(BRAIN_array)
            INFARCT_array = mask2binary(INFARCT_array)
            print(">>> Unique Value of BRAIN mask :", np.unique(BRAIN_array))
            print(">>> Unique Value of INFARCT mask :", np.unique(INFARCT_array))

            # Pre-processing (2)- Normalization
            if norm == 'new':
                FLAIR_array = ws_normalize(FLAIR_array, 'FLAIR', BRAIN_array)
                
                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : White-stripe Normalization ({})".format(time))
                
            elif norm == 'ws':
                FLAIR_array = normalization(FLAIR_array, INFARCT_array)

                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : New Normalization ({})".format(time))
                
            else:
                print("Value of 'norm' parameter should be 'new' of 'ws'")
                raise ValueError


            # Feature Extraction by Radiomics
            ADC_values, ADC_columns = feature_extract(ADC_array, INFARCT_array, features)
            FLAIR_values, FLAIR_columns = feature_extract(FLAIR_array, INFARCT_array, features)

            total_values = ADC_values + FLAIR_values
            # total_columns = ['ADC_' + col for col in ADC_columns] + ['FLAIR_' + col for col in ADC_columns]

            X.append(total_values)
            y.append(1)
            patient_num.append(pos_patient)


        except Exception as ex:
            time = str(datetime.datetime.now()).split()[1].split('.')[0]
            print("Error !!! [Patient Number : {}] ({})".format(i + 1, time))
            print(ex)

    for i, neg_patient in enumerate(neg_patient_list):
        time = str(datetime.datetime.now()).split()[1].split('.')[0]
        print("Processing [{0}/{1}] Image of Negative Patient... ({2})".format(i + 1, len(neg_patient_list), time))

        ADC_path, FLAIR_path, b1000_path, BRAIN_path, INFARCT_path = sorted(
            [path for path in neg_file_list if neg_patient in path])

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

                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : Voxel Size Resampling ({})".format(time))

            # Make Mask to Binary (0 or 1)
            BRAIN_array = mask2binary(BRAIN_array)
            INFARCT_array = mask2binary(INFARCT_array)
            print(">>> Unique Value of BRAIN mask :", np.unique(BRAIN_array))
            print(">>> Unique Value of INFARCT mask :", np.unique(INFARCT_array))

            # Pre-processing (2)- Normalization
            if norm == 'new':
                FLAIR_array = ws_normalize(FLAIR_array, 'FLAIR', BRAIN_array)
                
                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : White-stripe Normalization ({})".format(time))
                
            elif norm == 'ws':
                FLAIR_array = normalization(FLAIR_array, INFARCT_array)

                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : New Normalization ({})".format(time))
                
            else:
                print("Value of 'norm' parameter should be 'new' of 'ws'")
                raise ValueError

            # Feature Extraction by Radiomics
            ADC_values, ADC_columns = feature_extract(ADC_array, INFARCT_array, features)
            FLAIR_values, FLAIR_columns = feature_extract(FLAIR_array, INFARCT_array, features)

            total_values = ADC_values + FLAIR_values
            # total_columns = ['ADC_' + col for col in ADC_columns] + ['FLAIR_' + col for col in ADC_columns]

            X.append(total_values)
            y.append(0)
            patient_num.append(neg_patient)


        except Exception as ex:
            time = str(datetime.datetime.now()).split()[1].split('.')[0]
            print("Error !!! [Patient Number : {}] ({})".format(i + 1, time))
            print(ex)

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

    return X, y



# Feature Extraction for Inference
def test_data_loader(test_dir='/data/test/', norm, do_resample=True, features = ['firstorder', 'shape'], target_voxel = (0.65, 0.65, 3)):
    # File List
    test_file_list = glob(test_dir + "*")
    test_patient_list = list(set([path.split('_')[0] for path in os.listdir(test_dir)]))

    # Data Container
    X = []
    patient_num = []
    error_patient = []

    for i, test_patient in enumerate(test_patient_list):

        try:
            time = str(datetime.datetime.now()).split()[1].split('.')[0]
            print("Processing [{0}/{1}] Image of Test Patient... ({2})".format(i + 1, len(test_patient_list), time))

            ADC_path, FLAIR_path, b1000_path, BRAIN_path, INFARCT_path = sorted(
                [path for path in test_file_list if test_patient in path])

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

                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : Voxel Size Resampling ({})".format(time))

            # Make Mask to Binary (0 or 1)
            BRAIN_array = mask2binary(BRAIN_array)
            INFARCT_array = mask2binary(INFARCT_array)
            print(">>> Unique Value of BRAIN mask :", np.unique(BRAIN_array))
            print(">>>Unique Value of INFARCT mask :", np.unique(INFARCT_array))

            # Pre-processing (2)- Normalization
            if norm == 'new':
                FLAIR_array = ws_normalize(FLAIR_array, 'FLAIR', BRAIN_array)
                
                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : White-stripe Normalization ({})".format(time))
                
            elif norm == 'ws':
                FLAIR_array = normalization(FLAIR_array, INFARCT_array)

                time = str(datetime.datetime.now()).split()[1].split('.')[0]
                print(">>> Finished : New Normalization ({})".format(time))
                
            else:
                print("Value of 'norm' parameter should be 'new' of 'ws'")
                raise ValueError

            # Feature Extraction by Radiomics
            ADC_values, ADC_columns = feature_extract(ADC_array, INFARCT_array, features)
            FLAIR_values, FLAIR_columns = feature_extract(FLAIR_array, INFARCT_array, features)

            total_values = ADC_values + FLAIR_values
            # total_columns = ['ADC_' + col for col in ADC_columns] + ['FLAIR_' + col for col in ADC_columns]

            X.append(total_values)
            patient_num.append(test_patient)

        except Exception as e:
            print("!! Error in", test_patient)
            print(e)

            error_patient.append(test_patient)

    X = np.array(X)
    
    return X, patient_num, error_patient
