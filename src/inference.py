from utils.data_loader import test_data_loader
from utils.inference_tools import pred_to_binary, export_csv

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import pandas as pd
import pickle
import datetime

import warnings
warnings.filterwarnings('ignore')


# Setting
test_dir = '/data/test/'

do_n4 = False
do_ws = True
do_resample = True

do_shuffle = False
save_to_disk = False
return_patient_num = True


# Data Load
X_test, patient_num = test_data_loader(test_dir, do_n4, do_ws, do_resample, do_shuffle, save_to_disk, return_patient_num)


#########################################################################################################################
#########################################################################################################################
#### Modify here ####


# Load trained model
threshold = 0.65
model = pickle.load(open('/data/model/model.pickle.dat', 'rb'))
print("\nStart Inference...")
print("Threshold :", threshold)


# Make Predictions for Test Data
y_pred = model.predict_proba(X_test)[:, 1]
y_pred_binary = pred_to_binary(y_pred, threshold = threshold = threshold)


# Make 'output.csv'
export_csv(patient_num, y_pred_binary, y_pred)
