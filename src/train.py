import numpy as np
from utils.data_loader import train_data_loader
from utils import utils
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearSVC
from sklearn.lda import LDA
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
name = 'Semin #1'
model = 'XGBoost + Logistic Regression + LDA + SVM | RandomSearchCV'
summary = '''Experiment: Ensemble a subset of {XGBoost} + {Logistic Regression, LDA, SVM}'''


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


# ---------------------- Model parameters definition ----------------------- #
xgb_params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

# rf_params = {
#     'n_estimators': [10,50,100],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 5, 10, 20],
#     'min_samples_split': [2, 3, 5, 10],
#     'max_features': ['sqrt', 'log2']
# }


# --------------------------- Model definitions ---------------------------- #

xgb_base = xgb.XGBClassifier(learning_rate=0.01, n_estimators=600, objective='binary:logistic', silent=True, nthread=1)


xgb_model = RandomizedSearchCV(xgb_base, xgb_params, cv=5, scoring=make_scorer(fbeta_score, beta=0.5),
                               verbose=3, n_jobs=-1)
lda_model = LDA()
logistic = LogisticRegression()
sgd = SGDClassifier()
svc = LinearSVC()


# ---------------------------- Model fitting ------------------------------- #
xgb_model.fit(X_train, y_train)
lda_model.fit(X_train, y_train)
logistic.fit(X_train, y_train)
sgd.fit(X_train, y_train)
svc.fit(X_train, y_train)


print("Base model: XGBoost")
print("Best params:", xgb_model.best_params_)
print("Best estimator:\n", xgb_model.best_estimator_)
print()

print("Rest of the models:")
print(lda_model)
print(logistic)
print(sgd)
print(svc)



# ----------------------------- Save model --------------------------------- #
pickle.dump(xgb_model, open('/data/model/xgb_model.pickle.dat', 'wb'))
pickle.dump(lda_model, open('/data/model/lda_model.pickle.dat', 'wb'))
pickle.dump(logistic, open('/data/model/logistic.pickle.dat', 'wb'))
pickle.dump(sgd, open('/data/model/sgd.pickle.dat', 'wb'))
pickle.dump(svc, open('/data/model/svc.pickle.dat', 'wb'))
