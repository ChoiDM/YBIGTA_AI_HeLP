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
from keras.layers import Dense
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
BETA2=0.5
cv=5
threshold = "auto"
norm = 'ws'
include_model = [1,4,10,11,12]
include_model2 = [1,2,3,4]
include_model3 = []
final_idx = 3


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
print("\n---------- ML Model Load ----------")
model1 = pickle.load(open(path+'/model/model1.pickle.dat', 'rb'))
model2 = pickle.load(open(path+'/model/model2.pickle.dat', 'rb'))
model3 = pickle.load(open(path+'/model/model3.pickle.dat', 'rb'))
model4 = pickle.load(open(path+'/model/model4.pickle.dat', 'rb'))
model5 = pickle.load(open(path+'/model/model5.pickle.dat', 'rb'))
model6 = pickle.load(open(path+'/model/model6.pickle.dat', 'rb'))
model7 = pickle.load(open(path+'/model/model7.pickle.dat', 'rb'))
model8 = pickle.load(open(path+'/model/model8.pickle.dat', 'rb'))
model9 = pickle.load(open(path+'/model/model9.pickle.dat', 'rb'))
model10 = pickle.load(open(path+'/model/model10.pickle.dat', 'rb'))
model11 = pickle.load(open(path+'/model/model11.pickle.dat', 'rb'))
model12 = pickle.load(open(path+'/model/model12.pickle.dat', 'rb'))
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Load Stacking model 1
print("\n---------- Stacking Model Load 1 ----------")

meta_xgb = pickle.load(open(path+'/model/meta_xgb.pickle.dat', 'rb'))
meta_logistic = pickle.load(open(path+'/model/meta_logistic.pickle.dat', 'rb'))

with open(path+'/model/meta_NN.json', 'r') as f :
    meta_NN = model_from_json(f.read())
meta_NN.model.load_weights(path+'/model/meta_NN.h5')

with open(path+'/model/meta_weight.json', 'r') as f :
    meta_weight = model_from_json(f.read())
meta_weight.model.load_weights(path+'/model/meta_weight.h5')
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Stacking model
print("\n---------- Inference ----------")
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]
models2 = [meta_xgb, meta_logistic, meta_NN, meta_weight]
models3 = []

# Layer1
print("\n---------- Layer1 ----------")
S_models = get_stacking_base_model(models, include_model)
scorer = make_scorer(fbeta_score, beta=BETA2)
S_train, S_test = vecstack.stacking(S_models, X_train, y_train, X_test, regression = False, metric=scorer, n_folds=cv, needs_proba=True)

y_pred_lst = []
y_pred_binary_lst =[]
y_pred_lst2 = []
y_pred_binary_lst2 =[]

for meta in models2 :
    pred = meta.predict_proba(S_test)[:, 1]
    y_pred_lst.append(pred)
    y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold))

    
# Make 'output.csv'
final, final_df = export_csv(patient_num, error_patient, y_pred_binary_lst, y_pred_lst, path = path, index=final_idx)
print("\n")
print(making_result(S_test, y_pred_lst, y_pred_binary_lst, y_pred_lst2, y_pred_binary_lst2, include_model, include_model2, include_model3, final))

print("\n\n\n")
print("----------------------------")
print("---------- Result ----------")
print("----------------------------")
print(final_df)
#------------------------------------------------------------------------------------------------------------------------

print("\n---------- inference.py finished ----------")

