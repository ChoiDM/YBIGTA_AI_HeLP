from utils.data_loader import test_data_loader
from utils.inference_tools import pred_to_binary, export_csv
from utils import utils
import pandas as pd
import xgboost as xgb
import pickle

print("File: {} | Start:".format(__file__), utils.now())

# Setting
test_dir = '/data/test/'

features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)

norm = 'new' # 'norm' should be 'ws' or 'new'
do_resample = True

threshold = 0.70
class_of_error_patient = 0


# Data Load
X_test, patient_num, error_patient = test_data_loader(test_dir, norm, do_resample, features, target_voxel)

# ---------------------------------------- Modify here ---------------------------------------- #


# Load trained model
# xgb = pickle.load(open('/data/model/xgb_model.pickle.dat', 'rb'))
# log = pickle.load(open('/data/model/logistic.pickle.dat', 'rb'))
# sgd = pickle.load(open('/data/model/sgd.pickle.dat', 'rb'))
# svc = pickle.load(open('/data/model/svc.pickle.dat', 'rb'))
final_model = pickle.load(open('/data/model/final_model.pickle.dat', 'rb'))

# Make Predictions for Test Data
# y_pred_xgb = xgb.predict_proba(X_test)[:, 1]
# y_pred_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred = y_pred_binary = final_model.predict(X_test)  # [:, 1]
# y_pred_binary = pred_to_binary(y_pred, threshold = 0.5)


# Make 'output.csv'
export_csv(patient_num, error_patient, class_of_error_patient, y_pred_binary, y_pred)
