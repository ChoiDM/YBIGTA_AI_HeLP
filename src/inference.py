from utils.data_loader import test_data_loader, data_generator
from utils.inference_tools import pred_to_binary, export_csv, making_result,  error_check
from utils.model_stacking import *
from utils.cube_tools import *
from utils.model_ml import *
import vecstack
from glob import glob

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, SGDClassifier, Lars, LassoLars
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Conv3D, Flatten, pooling
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
final_idx=2

batch_size = 4
mode="test"
cube_shape = (32, 32, 16)


# Data Load
print("---------- Start ----------")
print("\n---------- Data Load ----------")
features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)
do_resample = True
do_minmax = True

X_test, patient_num, error_patient = test_data_loader(test_dir, norm, do_resample, do_minmax, features, target_voxel, path=path)


#########################################################################################################################
#########################################################################################################################
#### Modify here ####

#------------------------------------------------------------------------------------------------------------------------
# Load trained model
print("\n---------- Model Load ----------")
with open(path+'/model/MLP.json', 'r') as f :
    MLP = model_from_json(f.read())
MLP.model.load_weights(path+'/model/MLP.h5')

with open(path+'/model/CNN.json', 'r') as f :
    CNN = model_from_json(f.read())
CNN.model.load_weights(path+'/model/CNN.h5')
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Stacking model
print("\n---------- Inference ----------")
y_pred_lst = []
y_pred_binary_lst =[]

## MLP
pred = MLP.predict_proba(X_test)[:, 1]
y_pred_lst.append(pred)
y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold))

## CNN
data_dir = sorted(glob(os.path.join(path, mode, '*')))
data_dir, error_patient = error_check(data_dir)
data_gen = data_generator(batch_size, mode, data_dir, cube_shape, norm, target_voxel)
    
pred = CNN.predict_generator(data_gen, steps=255//batch_size)
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

