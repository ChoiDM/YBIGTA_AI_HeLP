import numpy as np
from utils.data_loader import train_data_loader
from utils import utils
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer
import pickle
import datetime
import pprint

import warnings
warnings.filterwarnings('ignore')

# Print Current Time
print("File: train.py | Start:", utils.now())


# Print Information
name = 'Semin'
model = 'XGBoost + GridSearchCV'
summary = 'Boosting with RandomizedSearchCV'


print('---------------------------')
print('Author Name :', name)
print('Model :', model)
print('Summary :', summary)
print('---------------------------')


# Setting
pos_dir = "/data/train/positive/"
neg_dir = "/data/train/negative/"

do_n4 = False
do_ws = True
do_resample = True

do_shuffle = True
save_to_disk = False
return_patient_num = False


# ------------------------------------------ Modify here ------------------------------------------------ #


# Data Load
X_train, y_train = train_data_loader(pos_dir, neg_dir, do_n4, do_ws, do_resample, do_shuffle, save_to_disk, return_patient_num)


# Fit Model with Training Data
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

def f0_5(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return fbeta_score(y_test, y_pred, 0.5)

base = xgb.XGBClassifier(learning_rate=0.01, n_estimators=600, objective='binary:logistic', silent=True, nthread=1)
model = RandomizedSearchCV(base, params, cv=5, scoring=make_scorer(fbeta_score(beta = 0.5)),
                           verbose=3, n_jobs=-1)
model.fit(X_train, y_train)


print("Base model:", model.__class__.__name__)
print("Parameters:", params)


print("Best params:", model.best_params_)
print("Best estimator:\n", model.best_estimator_)

cvres = model.cv_results_
# for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(np.sqrt(-mean_score), params)

# Save model to file
pickle.dump(model, open('/data/model/model.pickle.dat', 'wb'))
