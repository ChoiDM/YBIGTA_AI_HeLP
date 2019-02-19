from utils.data_loader import train_data_loader, test_data_loader
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


# Print Current Time
time = str(datetime.datetime.now()).split()[1].split('.')[0]
print("---------- Start ----------")
print("Start:", time, "\n")


# Print Information
name = 'KHW'
model = 'ML Stacking'
summary = 'HyperParams tuning with 4 sklearn models + 4 stacking model  + 1 stacking model(NN) + BETA=0.75 + cv=5'

print('Author Name :', name)
print('Model :', model)
print('Summary :', summary)


# Setting
path = "/data"
pos_dir = path+"/train/positive/"
neg_dir = path+"/train/negative/"

features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)

do_n4 = False
do_ws = True
do_resample = True
do_shuffle = True


# Data Load
print("\n---------- Data Load ----------")
X_train, y_train = train_data_loader(pos_dir, neg_dir, do_n4, do_ws, do_resample, do_shuffle, features, target_voxel)


#########################################################################################################################
#########################################################################################################################
#### Modify here ####

#------------------------------------------------------------------------------------------------------------------------
# Fit ML model with training data
print("\n---------- Start ML Train ----------")
cv=5
BETA=0.75

#########
## scorer
def new_scorer(y_true, y_pred, threshold=0.5) :
    result = []
    global BETA

    for pred in list(y_pred) :
        if pred >= threshold :
            result.append(1)
        else :
            result.append(0)
            
    return fbeta_score(y_true, np.array(result), beta=BETA)

scorer = make_scorer(fbeta_score, beta=BETA)

#########
## model1
print("model1")
model1 = XGBClassifier()

m1_params1 = {'subsample': [0.6], 'colsample_bytree': [0.6], 'min_child_weight': [0.5], 'probability': [True], 
              'gamma': [3.0], 'n_estimators': [300], 'learning_rate': [0.01], 'max_depth': [7]}

"""
m1_params1 = {
    'max_depth' : [5,6,7,8],
    'min_child_weight' : [0.5, 1, 5, 10, 15, 20],
    'gamma' : [1.5, 2, 2.5, 3.0, 5],
    'subsample' : [0.5, 0.6, 0.8, 1.0],
    'colsample_bytree' : [0.5, 0.6, 0.8, 1.0],
    'probability' : [True],
    'learning_rate' : [0.01, 0.05, 0.1],
    'n_estimators' : [300, 500, 700]}
"""
m1_grid_1 = GridSearchCV(model1, param_grid=m1_params1, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
m1_grid_1.fit(X_train, y_train)

best_model1 = m1_grid_1.best_estimator_

print("Best Score : {}".format(m1_grid_1.best_score_))
print("Best Params : {}".format(m1_grid_1.best_params_))

#########
## model2
print("\nmodel2")
model2 = SVC()

m2_params1 = {'probability': [True], 'degree': [2], 'C': [0.001], 'gamma': [0.001]}

"""
m2_params1 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'gamma' : [0.001, 0.01, 0.1, 1, 2, 5, 10, 20],
    'degree' : [2,3,4],
    'probability' : [True]}
"""

m2_grid_1 = GridSearchCV(model2, param_grid=m2_params1, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
m2_grid_1.fit(X_train, y_train)

best_model2 = m2_grid_1.best_estimator_

print("Best Score : {}".format(m2_grid_1.best_score_))
print("Best Params : {}".format(m2_grid_1.best_params_))

#########
## model3
print("\nmodel3")
model3 = LogisticRegression()

m3_params1 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter' : [n for n in range(100,1101, 200)]}

m3_grid_1 = GridSearchCV(model3, param_grid=m3_params1, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
m3_grid_1.fit(X_train, y_train)

best_model3 = m3_grid_1.best_estimator_

print("Best Score : {}".format(m3_grid_1.best_score_))
print("Best Params : {}".format(m3_grid_1.best_params_))

#########
## model4
print("\nmodel4")
model4 = RandomForestClassifier()

m4_params1 = {'n_estimators': [500], 'min_samples_leaf': [50], 'max_depth': [15]}

"""
m4_params1 = {
    'max_depth' : [6, 8, 10, 15, 20, 30, 40, 50],
    'min_samples_leaf': [1, 2, 3, 4, 5,10, 20, 50],
    'n_estimators' : [100, 300, 500]}
"""

m4_grid_1 = GridSearchCV(model4, param_grid=m4_params1, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
m4_grid_1.fit(X_train, y_train)

best_model4 = m4_grid_1.best_estimator_

print("Best Score : {}".format(m4_grid_1.best_score_))
print("Best Params : {}".format(m4_grid_1.best_params_))

#########
## model5
print("\nmodel5")
model5 = LogisticRegression()

m5_params1 = {'penalty': ['l1'], 'C': [1], 'max_iter': [900]}

"""
m5_params1 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter' : [n for n in range(100,1101, 200)],
    'penalty' : ["l1"]}
"""

m5_grid_1 = GridSearchCV(model5, param_grid=m5_params1, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
m5_grid_1.fit(X_train, y_train)

best_model5 = m5_grid_1.best_estimator_

print("Best Score : {}".format(m5_grid_1.best_score_))
print("Best Params : {}".format(m5_grid_1.best_params_))

#########
## model6
print("\nmodel6")
model6 = RidgeClassifier()

m6_params1 = {'alpha': [10], 'max_iter': [None]}

"""
m6_params1 = {
    'alpha': [0.1, 1, 2, 5, 10, 20, 50, 100],
    'max_iter' : [None]+[n for n in range(100,1101, 200)]}
"""

m6_grid_1 = GridSearchCV(model6, param_grid=m6_params1, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
m6_grid_1.fit(X_train, y_train)

best_model6 = m6_grid_1.best_estimator_

print("Best Score : {}".format(m6_grid_1.best_score_))
print("Best Params : {}".format(m6_grid_1.best_params_))

#########
## model7
print("\nmodel7")
model7 = SGDClassifier()

m7_params1 = {'penalty': ['elasticnet'], 'loss': ['log'], 'alpha': [100], 'l1_ratio': [0.5], 'max_iter': [1400]}

"""
m7_params1 = {
    'alpha': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 50, 100],
    'l1_ratio':[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 
    'max_iter' : [None]+[n for n in range(800, 1601, 200)],
    'penalty' : ["elasticnet"],
    'loss' : ["log"]}
"""

m7_grid_1 = GridSearchCV(model7, param_grid=m7_params1, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
m7_grid_1.fit(X_train, y_train)

best_model7 = m7_grid_1.best_estimator_

print("Best Score : {}".format(m7_grid_1.best_score_))
print("Best Params : {}".format(m7_grid_1.best_params_))

#########
## model8
print("\nmodel8")
model8 = Lars()

m8_params1 = {'n_nonzero_coefs': [70]}

"""
m8_params1 = {
    'n_nonzero_coefs': [n for n in range(30, 150, 20)]}
"""

max_score=0
m8_best_t = 0
best_model8 = ""
m8_best_grid_1 = ""

for t in [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.45, 0.4, 0.45, 0.5, 0.6] :
    scorer2 = make_scorer(new_scorer, threshold=t)
    m8_grid_1 = GridSearchCV(model8, param_grid=m8_params1, scoring=scorer2, cv=cv, verbose=0, n_jobs=-1)
    m8_grid_1.fit(X_train, y_train)

    if max_score < m8_grid_1.best_score_ :
        best_model8 = m8_grid_1.best_estimator_
        m8_best_t = t
        m8_best_grid_1 = m8_grid_1
        
m8_grid_1 = m8_best_grid_1
best_model8 = m8_grid_1.best_estimator_

print("Best Score : {}".format(m8_grid_1.best_score_))     
print("Threshold :", m8_best_t)
print("Best Params : {}".format(m8_grid_1.best_params_))

#########
## model9
print("\nmodel9")
model9 = LassoLars()

m9_params1 = {'alpha': [0.1], 'max_iter': [800]}

"""
m9_params1 = {
    'alpha': [0.1, 1, 2, 5, 10, 20, 50, 100],
    'max_iter' : [n for n in range(800, 1601, 200)]}
"""

max_score=0
m9_best_t = 0
best_model9 = ""
m9_best_grid_1 = ""
for t in [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.45, 0.4, 0.45, 0.5, 0.6] :
    scorer2 = make_scorer(new_scorer, threshold=t)
    m9_grid_1 = GridSearchCV(model9, param_grid=m9_params1, scoring=scorer2, cv=cv, verbose=0, n_jobs=-1)
    m9_grid_1.fit(X_train, y_train)

    if max_score < m9_grid_1.best_score_ :
        best_model9 = m9_grid_1.best_estimator_
        m9_best_t = t
        m9_best_grid_1 = m9_grid_1

m9_grid_1 = m9_best_grid_1
best_model9 = m9_grid_1.best_estimator_

print("Best Score : {}".format(m9_grid_1.best_score_))     
print("Threshold :", m9_best_t)
print("Best Params : {}".format(m9_grid_1.best_params_))

##########
## model10
print("\nmodel10")
model10 = ExtraTreesClassifier()

m10_params1 = {'n_estimators': [50], 'max_depth': [3]}

"""
m10_params1 = {
    'max_depth' : [None, 3, 5, 7, 9],
    'n_estimators' : [10, 50, 100, 300, 500]}
"""

m10_grid_1 = GridSearchCV(model10, param_grid=m10_params1, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
m10_grid_1.fit(X_train, y_train)

best_model10 = m10_grid_1.best_estimator_

print("Best Score : {}".format(m10_grid_1.best_score_))
print("Best Params : {}".format(m10_grid_1.best_params_))
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Save ML model
print("\n---------- Save ML Model ----------")
pickle.dump(best_model1, open(path+'/model/model1.pickle.dat', 'wb'))
pickle.dump(best_model2, open(path+'/model/model2.pickle.dat', 'wb'))
pickle.dump(best_model3, open(path+'/model/model3.pickle.dat', 'wb'))
pickle.dump(best_model4, open(path+'/model/model4.pickle.dat', 'wb'))
pickle.dump(best_model5, open(path+'/model/model5.pickle.dat', 'wb'))
pickle.dump(best_model6, open(path+'/model/model6.pickle.dat', 'wb'))
pickle.dump(best_model7, open(path+'/model/model7.pickle.dat', 'wb'))
pickle.dump(best_model8, open(path+'/model/model8.pickle.dat', 'wb'))
pickle.dump(best_model9, open(path+'/model/model9.pickle.dat', 'wb'))
pickle.dump(best_model10, open(path+'/model/model10.pickle.dat', 'wb'))
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Fit stacking model
print("\n---------- Start Staking Train ----------")
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# Layer1
print("\n---------- Layer1 ----------")
models = [best_model1, best_model2, best_model3, best_model4, best_model5, best_model6, best_model7, best_model8, best_model9, best_model10]
S_train = stacking(models, X_train)

meta_xgb = stacking_xgb(S_train, y_train, cv=cv)
meta_logistic = stacking_logistic(S_train, y_train, cv=cv)
meta_NN = stacking_NN(S_train, y_train, cv=cv)
meta_weight = stacking_weight(S_train, y_train, cv=cv)

y_pred_lst = []
y_pred_binary_lst =[]
threshold = "auto"
for meta in [meta_xgb, meta_logistic, meta_NN, meta_weight] :
    pred = meta.predict_proba(S_train)[:, 1]
    y_pred_lst.append(pred)
    y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold))
    
# Layer2
print("\n---------- Layer2 ----------")
models2 = [meta_xgb, meta_logistic, meta_NN, meta_weight]
S_train2 = stacking(models2, S_train, layer=2)

meta_xgb2 = stacking_xgb(S_train2, y_train, cv=cv)
meta_logistic2 = stacking_logistic(S_train2, y_train, cv=cv)
meta_NN2 = stacking_NN(S_train2, y_train, cv=cv)
meta_weight2 = stacking_weight(S_train2, y_train, cv=cv)

y_pred_lst2 = []
y_pred_binary_lst2 =[]
threshold = "auto"
for meta in [meta_xgb2, meta_logistic2, meta_NN2, meta_weight2] :
    pred = meta.predict_proba(S_train2)[:, 1]
    y_pred_lst2.append(pred)
    y_pred_binary_lst2.append(pred_to_binary(pred, threshold = threshold))
    
# Print result
print("\n")
print(making_result(S_train, y_pred_lst, y_pred_binary_lst, y_pred_lst2, y_pred_binary_lst2, y_train))

# Select model
print("\n---------- Select Meta Model For Layer2 ----------")
meta_model2 = meta_weight2
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Save stacking model
print("\n---------- Save Staking Model ----------")

pickle.dump(meta_xgb, open(path+'/model/meta_xgb.pickle.dat', 'wb'))
pickle.dump(meta_logistic, open(path+'/model/meta_logistic.pickle.dat', 'wb'))

meta_NN.model.save_weights(path+'/model/meta_NN.h5')
with open(path+'/model/meta_NN.json', 'w') as f :
    f.write(meta_NN.model.to_json())
    
meta_weight.model.save_weights(path+'/model/meta_weight.h5')
with open(path+'/model/meta_weight.json', 'w') as f :
    f.write(meta_weight.model.to_json())

print("\n---------- Save Staking Model2 ----------")
pickle.dump(meta_xgb2, open(path+'/model/meta_xgb2.pickle.dat', 'wb'))
pickle.dump(meta_logistic2, open(path+'/model/meta_logistic2.pickle.dat', 'wb'))

meta_NN2.model.save_weights(path+'/model/meta_NN2.h5')
with open(path+'/model/meta_NN2.json', 'w') as f :
    f.write(meta_NN2.model.to_json())
    
meta_weight2.model.save_weights(path+'/model/meta_weight2.h5')
with open(path+'/model/meta_weight2.json', 'w') as f :
    f.write(meta_weight2.model.to_json())
    
meta_model2.model.save_weights(path+'/model/meta_model2.h5')
with open(path+'/model/meta_model2.json', 'w') as f :
    f.write(meta_model2.model.to_json())
#------------------------------------------------------------------------------------------------------------------------
print("\n---------- train.py finished ----------")



