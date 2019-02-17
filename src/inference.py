from utils.data_loader import test_data_loader
from utils.inference_tools import pred_to_binary, export_csv
import pandas as pd
import xgboost as xgb
import pickle

import warnings
warnings.filterwarnings('ignore')


# Setting
test_dir = '/data/test/'

features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)

do_n4 = False
do_ws = True
do_resample = True

threshold = 0.629
class_of_error_patient = 0


# Data Load
X_test, patient_num, error_patient = test_data_loader(test_dir, do_n4, do_ws, do_resample, features, target_voxel)


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
