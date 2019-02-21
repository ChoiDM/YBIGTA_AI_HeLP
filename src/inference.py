from utils.data_loader import test_data_loader
from utils.inference_tools import pred_to_binary, export_csv
import pandas as pd
import xgboost as xgb
import pickle

import warnings
warnings.filterwarnings('ignore')


# Print Information
name = 'master-branch'
model = 'XGBoost'
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

threshold = 0.70
class_of_error_patient = 0


# Data Load
X_test, patient_num, error_patient = test_data_loader(test_dir, norm, do_resample, do_shuffle, features, target_voxel)


#########################################################################################################################
#########################################################################################################################
#### Modify here ####


# Load trained model
xgbClassifier = pickle.load(open('/data/model/xgb.pickle.dat', 'rb'))


# Make Predictions for Test Data
y_pred = xgbClassifier.predict_proba(X_test)[:, 1]
y_pred_binary = pred_to_binary(y_pred, threshold = threshold)


# Make 'output.csv'
export_csv(patient_num, error_patient, class_of_error_patient, y_pred_binary, y_pred)
