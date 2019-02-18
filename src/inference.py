from utils.data_loader import test_data_loader
from utils.inference_tools import pred_to_binary, export_csv, making_result
from utils.model_stacking import *

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, SGDClassifier, Lars, LassoLars
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

import pandas as pd
import numpy as np
import pickle
import datetime

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# Setting
path = "/data"
test_dir = path+'/test/'

features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)

do_n4 = False
do_ws = True
do_resample = True

# Data Load
print("\n---------- Data Load ----------")
X_test, patient_num, error_patient = test_data_loader(test_dir, do_n4, do_ws, do_resample, features, target_voxel)


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
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Load Stacking model
print("\n---------- Stacking Model Load1 ----------")
meta_xgb = pickle.load(open(path+'/model/meta_xgb.pickle.dat', 'rb'))
meta_logistic = pickle.load(open(path+'/model/meta_logistic.pickle.dat', 'rb'))

with open(path+'/model/meta_NN.json', 'r') as f :
    meta_NN = model_from_json(f.read())
meta_NN.model.load_weights(path+'/model/meta_NN.h5')

with open(path+'/model/meta_weight.json', 'r') as f :
    meta_weight = model_from_json(f.read())
meta_weight.model.load_weights(path+'/model/meta_weight.h5')

print("\n---------- Stacking Model Load1 ----------")
meta_xgb2 = pickle.load(open(path+'/model/meta_xgb2.pickle.dat', 'rb'))
meta_logistic2 = pickle.load(open(path+'/model/meta_logistic2.pickle.dat', 'rb'))

with open(path+'/model/meta_NN2.json', 'r') as f :
    meta_NN2 = model_from_json(f.read())
meta_NN2.model.load_weights(path+'/model/meta_NN2.h5')

with open(path+'/model/meta_weight2.json', 'r') as f :
    meta_weight2 = model_from_json(f.read())
meta_weight2.model.load_weights(path+'/model/meta_weight2.h5')
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Stacking model
print("\n---------- Model Stacking ----------")
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
models2 = [meta_xgb, meta_logistic, meta_NN, meta_weight]
models3 = [meta_xgb2, meta_logistic2, meta_NN2, meta_weight2]

threshold = "auto"
print("\n---------- Inference ----------")
print("Threshold :", threshold)

# Layer1
print("\n---------- Layer1 ----------")
S_test = stacking(models, X_test)
y_pred_lst = []
y_pred_binary_lst =[]
for meta in models2 :
    pred = meta.predict_proba(S_test)[:, 1]
    y_pred_lst.append(pred)
    y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold))

# Layer2
print("\n---------- Layer2 ----------")
S_test2 = stacking(models2, S_test, layer=2)
y_pred_lst2 = []
y_pred_binary_lst2 =[]
threshold = "auto"
for meta in models3 :
    pred = meta.predict_proba(S_test2)[:, 1]
    y_pred_lst2.append(pred)
    y_pred_binary_lst2.append(pred_to_binary(pred, threshold = threshold))

# Make 'output.csv'
final, final_df = export_csv(patient_num, error_patient, y_pred_binary_lst2, y_pred_lst2, path = path, index=3)
print(making_result(S_test, y_pred_lst, y_pred_binary_lst, y_pred_lst2, y_pred_binary_lst2, final))
print("\n\n\n----------------------------")
print("---------- Result ----------")
print("----------------------------")
print(final_df)
print("\n---------- inference.py finished ----------")

