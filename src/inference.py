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


print("---------- Start ----------")
path = "/data"
save_dir = path+"/model/"
test_dir = path+'/test/'


# Setting
# Set your params here!!!
threshold = "auto"
norm = 'new'
mode="test"
cube_shape = (32, 32, 16)
final_idx=2 # 1=MLP, 2=CNN ----------> check this parmas carefully!!


# Data Load
features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)
do_resample = True
do_minmax = True


#########################################################################################################################
#########################################################################################################################
#### Modify here ####

#------------------------------------------------------------------------------------------------------------------------
# Load trained model

if final_idx == 1 :
    print("\n---------- Data Load ----------")
    X_test, patient_num, error_patient = test_data_loader(test_dir, norm, do_resample, do_minmax, features, target_voxel, path=path)
    
    print("\n---------- Model Load ----------")
    with open(path+'/model/MLP.json', 'r') as f :
        MLP = model_from_json(f.read())
    MLP.model.load_weights(path+'/model/MLP.h5')
    
    y_pred_lst = []
    y_pred_binary_lst =[]
    
    print("\n---------- Inference ----------")
    pred = MLP.predict_proba(X_test)[:, 1]
    y_pred_lst.append(pred)
    y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold))
    
elif final_idx==2 :
    print("\n---------- Model Load ----------")
    with open(path+'/model/CNN.json', 'r') as f :
        CNN = model_from_json(f.read())
    CNN.model.load_weights(path+'/model/CNN.h5')
    
    y_pred_lst = []
    y_pred_binary_lst =[]
    
    print("\n---------- Data Load ----------")
    data_dir = sorted(glob(os.path.join(path, mode, '*')))
    data_dir, error_patient = error_check(data_dir)
    data_gen = data_generator(1, mode, data_dir, cube_shape, norm, target_voxel)

    print("\n---------- Inference ----------")
    pred = CNN.predict_generator(data_gen, steps=len(data_dir))
    y_pred_lst.append(pred)
    y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold)) 
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Make 'output.csv'
final, final_df = export_csv(patient_num, error_patient, y_pred_binary_lst, y_pred_lst, path = path, index=1)

print("\n----------------------------")
print("---------- Result ----------")
print("----------------------------")
print(final_df)
#------------------------------------------------------------------------------------------------------------------------

print("\n---------- inference.py finished ----------")

