from utils.cube_data_loader import test_data_loader
from utils.inference_tools import pred_to_binary, export_csv
import pandas as pd
import pickle
from keras.models import load_model
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# Print Information
name = 'taeoh_base'
model = 'CNN_base'
summary = 'Normalization with new method / Threshold to 0.70'

print('---------------------------')
print('Author Name :', name)
print('Model :', model)
print('Summary :', summary)
print('---------------------------')


# Setting
test_dir = '/data/test/'

features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)

norm = 'new' # 'norm' should be 'ws' or 'new'
do_resample = True
do_shuffle = False  # DO NOT CHANGE!
brain_mask=True

threshold = 0.70
class_of_error_patient = 0


# Data Load
X_ADC, X_FLAIR, patient_num, error_patient = test_data_loader(test_dir, norm, do_resample, do_shuffle, features, target_voxel, brain_mask)

X_test = np.hstack([X_ADC, X_FLAIR])
#########################################################################################################################
#########################################################################################################################
#### Modify here ####


# Load trained model
model = load_model('/data/model/model.h5')
print(type(model))

# Make Predictions for Test Data
# y_pred = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
y_pred_binary = pred_to_binary(y_pred, threshold = threshold)


# Make 'output.csv'
export_csv(patient_num, error_patient, class_of_error_patient, y_pred_binary, y_pred)
