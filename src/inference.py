from utils.data_loader import test_data_loader
from utils.inference_tools import pred_to_binary, export_csv

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
test_dir = '/data/test/'

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


# Load trained model
print("\n---------- ML Model Load ----------")
model1 = pickle.load(open('/data/model/model1.pickle.dat', 'rb'))
model2 = pickle.load(open('/data/model/model2.pickle.dat', 'rb'))
model3 = pickle.load(open('/data/model/model3.pickle.dat', 'rb'))
model4 = pickle.load(open('/data/model/model4.pickle.dat', 'rb'))
model5 = pickle.load(open('/data/model/model5.pickle.dat', 'rb'))
model6 = pickle.load(open('/data/model/model6.pickle.dat', 'rb'))
model7 = pickle.load(open('/data/model/model7.pickle.dat', 'rb'))
model8 = pickle.load(open('/data/model/model8.pickle.dat', 'rb'))
model9 = pickle.load(open('/data/model/model9.pickle.dat', 'rb'))
model10 = pickle.load(open('/data/model/model10.pickle.dat', 'rb'))


# Stacking model
print("\n---------- Model Stacking ----------")
def stacking(models, data) : 
    result = []
    
    for idx, model in enumerate(models) :
        if idx+1 in [2,9] :
            continue
        if idx+1 in [6,8] :
            result.append(model.predict(data))
        else :
            result.append(model.predict_proba(data)[:,1])
        print("model", idx+1, "is stacked")
        
    return np.array(result).T

models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
S_test = stacking(models, X_test)
print(S_test)


# Load stacking model
print("\n---------- Model Stacking ----------")
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

with open('/data/model/model_architecture.json', 'r') as f :
    meta = model_from_json(f.read())

meta.model.load_weights('/data/model/model_weights.h5')


# Make Predictions for Test Data
threshold = 0.6
print("\n---------- Inference ----------")
print("Threshold :", threshold)

y_pred = meta.predict_proba(S_test)[:, 1]
y_pred_binary = pred_to_binary(y_pred, threshold = threshold)
print(y_pred_binary)


# Make 'output.csv'
export_csv(patient_num, y_pred_binary, y_pred)

