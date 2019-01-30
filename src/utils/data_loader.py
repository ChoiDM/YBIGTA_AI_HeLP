from glob import glob
import nibabel as nib
import numpy as np
import pandas as pd
import os
from random import shuffle

from utils.N4Correction import n4correction
from utils.Resample import resample, mask2binary
from utils.WhiteStripeNormalization import ws_normalize
from utils.FeatureExtract import feature_extract

import warnings
warnings.filterwarnings('ignore')



# Feature Extraction for Train
def train_data_loader(pos_dir = '/data/train/positive/', neg_dir = '/data/train/negative/', do_n4 = True, do_ws = True, do_resample = True,
                do_shuffle = True, save_to_disk = False, return_patient_num = False):

    # File List
    pos_file_list = glob(pos_dir + "*")
    neg_file_list = glob(neg_dir + "*")

    pos_patient_list = list(set([path.split('_')[0] for path in os.listdir(pos_dir)]))
    neg_patient_list = list(set([path.split('_')[0] for path in os.listdir(neg_dir)]))

    # Data Container
    X = []
    y = []
    patient_num = []


    for i, pos_patient in enumerate(pos_patient_list):
        print("Processing [{0}/{1}] Image of Positive Patient...".format(i+1, len(pos_patient_list)))

        ADC_path, FLAIR_path, b1000_path, BRAIN_path, INFARCT_path = sorted([path for path in pos_file_list if pos_patient in path])


        # Data Preparing
        ADC_array = nib.load(ADC_path).get_data()
        FLAIR_array = nib.load(FLAIR_path).get_data()
        BRAIN_array = nib.load(BRAIN_path).get_data()

        INFARCT_nii = nib.load(INFARCT_path)
        INFARCT_array = INFARCT_nii.get_data()

        origin_voxel_size = INFARCT_nii.header.get_zooms()


        # Pre-processing (1)- Resampling Voxel Size
        if do_resample:
            ADC_array = resample(ADC_array, origin_voxel_size)
            FLAIR_array = resample(FLAIR_array, origin_voxel_size)
            BRAIN_array = resample(BRAIN_array, origin_voxel_size)
            INFARCT_array = resample(INFARCT_array, origin_voxel_size)

            print(">>> Finished : Voxel Size Resampling")


        # Make Mask to Binary (0 or 1)
        BRAIN_array = mask2binary(BRAIN_array)
        INFARCT_array = mask2binary(INFARCT_array)


        # Pre-processing (2)- N4 Bias Correction
        if do_n4:
            ADC_array = n4correction(ADC_array)
            FLAIR_array = n4correction(FLAIR_array)

            print(">>> Finished : N4 Bias Correction")


        # Pre-processing (3)- White-stripe Normalization
        if do_ws:
            ADC_array = ws_normalize(ADC_array, 'T2', BRAIN_array)
            FLAIR_array = ws_normalize(FLAIR_array, 'FLAIR', BRAIN_array)

            print(">>> Finished : White-stripe Normalization")

        # Feature Extraction by Radiomics
        ADC_values, ADC_columns = feature_extract(ADC_array, INFARCT_array)
        FLAIR_values, FLAIR_columns = feature_extract(FLAIR_array, INFARCT_array)

        total_values = ADC_values + FLAIR_values
        total_columns = ['ADC_' + col for col in ADC_columns] + ['FLAIR_' + col for col in ADC_columns]


        # Save
        if save_to_disk:

            if not os.path.exists("output"):
                os.mkdir("output")

            ADC_to_save = nib.Nifti1Image(ADC_array, np.eye(4))
            FLAIR_to_save = nib.Nifti1Image(FLAIR_array, np.eye(4))
            BRAIN_to_save = nib.Nifti1Image(BRAIN_array, np.eye(4))
            INFARCT_to_save = nib.Nifti1Image(INFARCT_array, np.eye(4))

            nib.save(ADC_to_save, os.path.join("output", pos_patient + "_ADC_resampled.nii"))
            nib.save(FLAIR_to_save, os.path.join("output", pos_patient + "_FLAIR_resampled.nii"))
            nib.save(BRAIN_to_save, os.path.join("output", pos_patient + "_BRAIN_resampled.nii"))
            nib.save(INFARCT_to_save, os.path.join("output", pos_patient + "_INFARCT_resampled.nii"))


            if not os.path.isfile("output/total_columns.txt"):

                with open("output/total_columns.txt", "w") as output_file:
                    for col in total_columns:
                        output_file.write(col + ' ')


            print(">>> Finished : Saved to Disk")


        X.append(total_values)
        y.append(1)
        patient_num.append(pos_patient)



    for i, neg_patient in enumerate(neg_patient_list):
        print("Processing [{0}/{1}] Image of Negative Patient...".format(i+1, len(neg_patient_list)))

        ADC_path, FLAIR_path, b1000_path, BRAIN_path, INFARCT_path = sorted([path for path in neg_file_list if neg_patient in path])


        # Data Preparing
        ADC_array = nib.load(ADC_path).get_data()
        FLAIR_array = nib.load(FLAIR_path).get_data()
        BRAIN_array = nib.load(BRAIN_path).get_data()

        INFARCT_nii = nib.load(INFARCT_path)
        INFARCT_array = INFARCT_nii.get_data()

        origin_voxel_size = INFARCT_nii.header.get_zooms()


        # Pre-processing (1)- Resampling Voxel Size
        if do_resample:
            ADC_array = resample(ADC_array, origin_voxel_size)
            FLAIR_array = resample(FLAIR_array, origin_voxel_size)
            BRAIN_array = resample(BRAIN_array, origin_voxel_size)
            INFARCT_array = resample(INFARCT_array, origin_voxel_size)

            print(">>> Finished : Voxel Size Resampling")


        # Make Mask to Binary (0 or 1)
        BRAIN_array = mask2binary(BRAIN_array)
        INFARCT_array = mask2binary(INFARCT_array)


        # Pre-processing (2)- N4 Bias Correction
        if do_n4:
            ADC_array = n4correction(ADC_array)
            FLAIR_array = n4correction(FLAIR_array)

            print(">>> Finished : N4 Bias Correction")


        # Pre-processing (3)- White-stripe Normalization
        if do_ws:
            ADC_array = ws_normalize(ADC_array, 'T2', BRAIN_array)
            FLAIR_array = ws_normalize(FLAIR_array, 'FLAIR', BRAIN_array)

            print(">>> Finished : White-stripe Normalization")

        # Feature Extraction by Radiomics
        ADC_values, ADC_columns = feature_extract(ADC_array, INFARCT_array)
        FLAIR_values, FLAIR_columns = feature_extract(FLAIR_array, INFARCT_array)

        total_values = ADC_values + FLAIR_values
        total_columns = ['ADC_' + col for col in ADC_columns] + ['FLAIR_' + col for col in ADC_columns]


        # Save
        if save_to_disk:

            if not os.path.exists("output"):
                os.mkdir("output")

            ADC_to_save = nib.Nifti1Image(ADC_array, np.eye(4))
            FLAIR_to_save = nib.Nifti1Image(FLAIR_array, np.eye(4))
            BRAIN_to_save = nib.Nifti1Image(BRAIN_array, np.eye(4))
            INFARCT_to_save = nib.Nifti1Image(INFARCT_array, np.eye(4))

            nib.save(ADC_to_save, os.path.join("output", neg_patient + "_ADC_resampled.nii"))
            nib.save(FLAIR_to_save, os.path.join("output", neg_patient + "_FLAIR_resampled.nii"))
            nib.save(BRAIN_to_save, os.path.join("output", neg_patient + "_BRAIN_resampled.nii"))
            nib.save(INFARCT_to_save, os.path.join("output", neg_patient + "_INFARCT_resampled.nii"))


            if not os.path.isfile("output/total_columns.txt"):

                with open("output/total_columns.txt", "w") as output_file:
                    for col in total_columns:
                        output_file.write(col + ' ')

            print(">>> Finished : Saved to Disk")


        X.append(total_values)
        y.append(0)
        patient_num.append(neg_patient)


    if do_shuffle:
        shuffle_list = [[X_value, y_value, num] for X_value, y_value, num in zip(X, y, patient_num)]
        shuffle(shuffle_list)

        X = [value[0] for value in shuffle_list]
        y = [value[1] for value in shuffle_list]
        patient_num = [value[2] for value in shuffle_list]


    X = np.array(X)
    y = np.array(y)



    if return_patient_num:
        return X, y, patient_num

    else:
        return X, y


# Feature Extraction for Inference
def test_data_loader(test_dir = '/data/test/', do_n4 = True, do_ws = True, do_resample = True,
                do_shuffle = True, save_to_disk = False, return_patient_num = False):

    # File List
    test_file_list = glob(test_dir + "*")
    test_patient_list = list(set([path.split('_')[0] for path in os.listdir(test_dir)]))

    # Data Container
    X = []
    patient_num = []


    for i, test_patient in enumerate(test_patient_list):
        print("Processing [{0}/{1}] Image of Positive Patient...".format(i+1, len(test_patient_list)))

        ADC_path, FLAIR_path, b1000_path, BRAIN_path, INFARCT_path = sorted([path for path in test_file_list if test_patient in path])


        # Data Preparing
        ADC_array = nib.load(ADC_path).get_data()
        FLAIR_array = nib.load(FLAIR_path).get_data()
        BRAIN_array = nib.load(BRAIN_path).get_data()

        INFARCT_nii = nib.load(INFARCT_path)
        INFARCT_array = INFARCT_nii.get_data()

        origin_voxel_size = INFARCT_nii.header.get_zooms()


        # Pre-processing (1)- Resampling Voxel Size
        if do_resample:
            ADC_array = resample(ADC_array, origin_voxel_size)
            FLAIR_array = resample(FLAIR_array, origin_voxel_size)
            BRAIN_array = resample(BRAIN_array, origin_voxel_size)
            INFARCT_array = resample(INFARCT_array, origin_voxel_size)

            print(">>> Finished : Voxel Size Resampling")


        # Make Mask to Binary (0 or 1)
        BRAIN_array = mask2binary(BRAIN_array)
        INFARCT_array = mask2binary(INFARCT_array)


        # Pre-processing (2)- N4 Bias Correction
        if do_n4:
            ADC_array = n4correction(ADC_array)
            FLAIR_array = n4correction(FLAIR_array)

            print(">>> Finished : N4 Bias Correction")


        # Pre-processing (3)- White-stripe Normalization
        if do_ws:
            ADC_array = ws_normalize(ADC_array, 'T2', BRAIN_array)
            FLAIR_array = ws_normalize(FLAIR_array, 'FLAIR', BRAIN_array)

            print(">>> Finished : White-stripe Normalization")


        # Feature Extraction by Radiomics
        ADC_values, ADC_columns = feature_extract(ADC_array, INFARCT_array)
        FLAIR_values, FLAIR_columns = feature_extract(FLAIR_array, INFARCT_array)

        total_values = ADC_values + FLAIR_values
        total_columns = ['ADC_' + col for col in ADC_columns] + ['FLAIR_' + col for col in ADC_columns]


        # Save
        if save_to_disk:

            if not os.path.exists("output"):
                os.mkdir("output")

            ADC_to_save = nib.Nifti1Image(ADC_array, np.eye(4))
            FLAIR_to_save = nib.Nifti1Image(FLAIR_array, np.eye(4))
            BRAIN_to_save = nib.Nifti1Image(BRAIN_array, np.eye(4))
            INFARCT_to_save = nib.Nifti1Image(INFARCT_array, np.eye(4))

            nib.save(ADC_to_save, os.path.join("output", test_patient + "_ADC_resampled.nii"))
            nib.save(FLAIR_to_save, os.path.join("output", test_patient + "_FLAIR_resampled.nii"))
            nib.save(BRAIN_to_save, os.path.join("output", test_patient + "_BRAIN_resampled.nii"))
            nib.save(INFARCT_to_save, os.path.join("output", test_patient + "_INFARCT_resampled.nii"))


            if not os.path.isfile("output/total_columns.txt"):

                with open("output/total_columns.txt", "w") as output_file:
                    for col in total_columns:
                        output_file.write(col + ' ')


            print(">>> Finished : Saved to Disk")


        X.append(total_values)
        patient_num.append(test_patient)


    if do_shuffle:
        shuffle_list = [[X_value, num] for X_value, num in zip(X, patient_num)]
        shuffle(shuffle_list)

        X = [value[0] for value in shuffle_list]
        patient_num = [value[1] for value in shuffle_list]

    X = np.array(X)

    if return_patient_num:
        return X, patient_num

    else:
        return X


