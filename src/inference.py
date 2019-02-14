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


# Setting
path = "/data"
test_dir = path+'/test/'

do_n4 = False
do_ws = True
do_resample = True

do_shuffle = False
save_to_disk = False
return_patient_num = True


# Data Load
print("\n---------- Data Load ----------")
X_test, patient_num = test_data_loader(test_dir, do_n4, do_ws, do_resample, do_shuffle, save_to_disk, return_patient_num)


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
model6 = pickle.load(open(path+'a/model/model6.pickle.dat', 'rb'))
model7 = pickle.load(open(path+'/model/model7.pickle.dat', 'rb'))
model8 = pickle.load(open(path+'/model/model8.pickle.dat', 'rb'))
model9 = pickle.load(open(path+'/model/model9.pickle.dat', 'rb'))
model10 = pickle.load(open(path+'/model/model10.pickle.dat', 'rb'))
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Stacking model
print("\n---------- Stacking Model Load ----------")
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
# Load stacking model
print("\n---------- Model Stacking ----------")
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
S_test = stacking(models, X_test)

# Make Predictions for Test Data
threshold = "auto"
print("\n---------- Inference ----------")
print("Threshold :", threshold)

y_pred_lst = []
y_pred_binary_lst =[]
for meta in [meta_xgb, meta_logistic, meta_NN, meta_weight] :
    y_pred_lst.append(meta.predict_proba(S_train)[:, 1])
    y_pred_binary_lst.append(pred_to_binary(y_pred_xgb, threshold = threshold))

# TODO : add one more model stacking

# Make 'output.csv'
final = export_csv(patient_num, y_pred_binary, y_pred, path = path, index=0)
print(making_result(S_test, y_pred_lst, y_pred_binary_lst, final))

