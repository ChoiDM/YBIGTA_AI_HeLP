from utils.data_loader import train_data_loader, test_data_loader
from utils.inference_tools import pred_to_binary, export_csv, making_result
from utils.model_stacking import *
from utils.model_ml import *

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, SGDClassifier, Lars, LassoLars
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from keras.models import Sequential
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


# Print Current Time
time = str(datetime.datetime.now()).split()[1].split('.')[0]
print("---------- Start ----------")
print("Start:", time, "\n")


# Print Information
name = 'KHW2'
model = 'ML Stacking'
summary = 'HyperParams tuning with 6 ML models + 4 stacking model  + 1 stacking model(NN) + BETA=0.75 + cv=5 + threshold=auto + Norm=new'

print('Author Name :', name)
print('Model :', model)
print('Summary :', summary)


# Setting
path = "/data"
pos_dir = path+"/train/positive/"
neg_dir = path+"/train/negative/"

features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)

norm = 'new'
do_resample = True
do_shuffle = True


# Data Load
print("\n---------- Data Load ----------")
X_train, y_train = train_data_loader(pos_dir, neg_dir, norm, do_resample, do_shuffle, features, target_voxel)


####################################################################z#####################################################
#########################################################################################################################
#### Modify here ####

#------------------------------------------------------------------------------------------------------------------------
# Fit ML model with training data
print("\n---------- Start ML Train ----------")
cv=5
BETA=0.75
BETA2=0.5
threshold = "auto"

#########
## scorer
scorer = make_scorer(fbeta_score, beta=BETA)

#########
## model1
print("model1")
m1_params = {'subsample': [0.6], 'colsample_bytree': [0.6], 'min_child_weight': [0.5], 'probability': [True], 
              'gamma': [3.0], 'n_estimators': [300], 'learning_rate': [0.01], 'max_depth': [7]}
model1 = ml_xgb(X_train, y_train, cv=cv, beta=BETA, params=m1_params)

#########
## model2
print("\nmodel2")
m2_params = {'probability': [True], 'degree': [2], 'C': [0.001], 'gamma': [0.001]}
model2 = ml_svm(X_train, y_train, cv=cv, beta=BETA, params=m2_params)

#########
## model3
print("\nmodel3")
model3 = ml_logistic(X_train, y_train, cv=cv, beta=BETA)

#########
## model4
print("\nmodel4")
m4_params = {'n_estimators': [500], 'min_samples_leaf': [50], 'max_depth': [15]}
model4 = ml_rf(X_train, y_train, cv=cv, beta=BETA, params=m4_params)

#########
## model5
print("\nmodel5")
m5_params = {'penalty': ['l1'], 'C': [1], 'max_iter': [900]}
model5 = ml_lasso(X_train, y_train, cv=cv, beta=BETA, params=m5_params)

#########
## model6
print("\nmodel6")
m6_params =  {'alpha': [10], 'max_iter': [None]}
model6 = ml_ridge(X_train, y_train, cv=cv, beta=BETA, params=m6_params)

#########
## model7
print("\nmodel7")
m7_params =  {'penalty': ['elasticnet'], 'loss': ['log'], 'alpha': [100], 'l1_ratio': [0.5], 'max_iter': [1400]}
model7 = ml_elasticNet(X_train, y_train, cv=cv, beta=BETA, params=m7_params)

#########
## model8
print("\nmodel8")
m8_params =  {'n_nonzero_coefs': [70]}
model8 = ml_lars(X_train, y_train, cv=cv, beta=BETA, params=m8_params)

#########
## model9
print("\nmodel9")
m9_params =  {'alpha': [0.1], 'max_iter': [800]}
model9 = ml_larsLasso(X_train, y_train, cv=cv, beta=BETA, params=m9_params)

##########
## model10
print("\nmodel10")
m10_params =  {'n_estimators': [50], 'max_depth': [3]}
model10 = ml_extraTrees(X_train, y_train, cv=cv, beta=BETA, params=m10_params)

##########
## model11
print("\nmodel11")
model11 = ml_adaboost(X_train, y_train, cv=cv, beta=BETA)

##########
## model10
print("\nmodel12")
model12 = ml_lightgbm(X_train, y_train, cv=cv, beta=BETA)
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Save ML model
print("\n---------- Save ML Model ----------")
pickle.dump(model1, open(path+'/model/model1.pickle.dat', 'wb'))
pickle.dump(model2, open(path+'/model/model2.pickle.dat', 'wb'))
pickle.dump(model3, open(path+'/model/model3.pickle.dat', 'wb'))
pickle.dump(model4, open(path+'/model/model4.pickle.dat', 'wb'))
pickle.dump(model5, open(path+'/model/model5.pickle.dat', 'wb'))
pickle.dump(model6, open(path+'/model/model6.pickle.dat', 'wb'))
pickle.dump(model7, open(path+'/model/model7.pickle.dat', 'wb'))
pickle.dump(model8, open(path+'/model/model8.pickle.dat', 'wb'))
pickle.dump(model9, open(path+'/model/model9.pickle.dat', 'wb'))
pickle.dump(model10, open(path+'/model/model10.pickle.dat', 'wb'))
pickle.dump(model11, open(path+'/model/model11.pickle.dat', 'wb'))
pickle.dump(model12, open(path+'/model/model12.pickle.dat', 'wb'))
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Fit stacking model
print("\n---------- Start Staking Train ----------")


# Layer1
print("\n---------- Layer1 ----------")
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]
include_model = [1,3,4,10,11,12]

S_train = stacking(models, X_train, include_model)

meta_xgb = stacking_xgb(S_train, y_train, cv=cv, beta=BETA2)
meta_logistic = stacking_logistic(S_train, y_train, cv=cv, beta=BETA2)
meta_NN = stacking_NN(S_train, y_train, cv=cv)
meta_weight = stacking_weight(S_train, y_train, cv=cv)

y_pred_lst = []
y_pred_binary_lst =[]
for meta in [meta_xgb, meta_logistic, meta_NN, meta_weight] :
    pred = meta.predict_proba(S_train)[:, 1]
    y_pred_lst.append(pred)
    y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold))
    
    
# Layer2
print("\n---------- Layer2 ----------")
models2 = [meta_xgb, meta_logistic, meta_NN, meta_weight]
include_model2 = [1,2,3,4]

S_train2 = stacking(models2, S_train, include_model2)

meta_NN2 = stacking_NN(S_train2, y_train, cv=cv)
meta_weight2 = stacking_weight(S_train2, y_train, cv=cv)

y_pred_lst2 = []
y_pred_binary_lst2 =[]
for meta in [meta_NN2, meta_weight2] :
    pred = meta.predict_proba(S_train2)[:, 1]
    y_pred_lst2.append(pred)
    y_pred_binary_lst2.append(pred_to_binary(pred, threshold = threshold))
    
    
# Print result
print("\n")
print(making_result(S_train, y_pred_lst, y_pred_binary_lst, y_pred_lst2, y_pred_binary_lst2, y_train, include_model, include_model2))
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Save stacking model 1
print("\n---------- Save Staking Model ----------")

pickle.dump(meta_xgb, open(path+'/model/meta_xgb.pickle.dat', 'wb'))
pickle.dump(meta_logistic, open(path+'/model/meta_logistic.pickle.dat', 'wb'))

meta_NN.model.save_weights(path+'/model/meta_NN.h5')
with open(path+'/model/meta_NN.json', 'w') as f :
    f.write(meta_NN.model.to_json())
    
meta_weight.model.save_weights(path+'/model/meta_weight.h5')
with open(path+'/model/meta_weight.json', 'w') as f :
    f.write(meta_weight.model.to_json())

    
# Save stacking model 2
print("\n---------- Save Staking Model2 ----------")

meta_NN2.model.save_weights(path+'/model/meta_NN2.h5')
with open(path+'/model/meta_NN2.json', 'w') as f :
    f.write(meta_NN2.model.to_json())
    
meta_weight2.model.save_weights(path+'/model/meta_weight2.h5')
with open(path+'/model/meta_weight2.json', 'w') as f :
    f.write(meta_weight2.model.to_json())
#------------------------------------------------------------------------------------------------------------------------

print("\n---------- train.py finished ----------")



