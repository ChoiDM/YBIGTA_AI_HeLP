from utils.data_loader import test_data_loader
from utils.inference_tools import pred_to_binary, export_csv, making_result
from utils.model_stacking import *
import vecstack

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, SGDClassifier, Lars, LassoLars
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

import pandas as pd
import numpy as np
import pickle
import datetime

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

path = "/data"
save_dir = path+"/model/"
test_dir = path+'/test/'


# Setting
# Set your params here!!!
threshold = "auto"
norm = 'new'
final_idx=1


# Data Load
print("---------- Start ----------")
print("\n---------- Data Load ----------")
features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)
do_resample = True

X_test, patient_num, error_patient = test_data_loader(test_dir, norm, do_resample, features, target_voxel)
X_train = np.load(save_dir+"X_train.npy")
y_train = np.load(save_dir+"y_train.npy")


#########################################################################################################################
#########################################################################################################################
#### Modify here ####

#------------------------------------------------------------------------------------------------------------------------
# Load trained model
print("\n---------- Model Load ----------")
with open(path+'/model/MLP.json', 'r') as f :
    MLP = model_from_json(f.read())
MLP.model.load_weights(path+'/model/MLP.h5')
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Stacking model
print("\n---------- Inference ----------")
models = [MLP]

y_pred_lst = []
y_pred_binary_lst =[]

for model in models :
    pred = model.predict_proba(X_test)[:, 1]
    y_pred_lst.append(pred)
    y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold))

    
# Make 'output.csv'
final, final_df = export_csv(patient_num, error_patient, y_pred_binary_lst, y_pred_lst, path = path, index=final_idx)

print("----------------------------")
print("---------- Result ----------")
print("----------------------------")
print(final_df)
#------------------------------------------------------------------------------------------------------------------------

print("\n---------- inference.py finished ----------")

