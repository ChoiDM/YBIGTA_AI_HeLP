import numpy as np
from utils.data_loader import train_data_loader
from utils import utils
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer

import pandas as pd
import pickle
import datetime
import pprint

import warnings
warnings.filterwarnings('ignore')

# Print Current Time
print("File: {} | Start:".format(__file__), utils.now())


# Print Information
name = 'Semin #2'
model = 'Hyper ensemble with XGBoost + Logistic Regression + SVM | RandomSearchCV'
summary = '''Hyper Ensemble {XGBoost, Logistic Regression, LDA, SVM} + data augmentation through horizontal flipping.
Uses approx. 50 models'''


print('---------------------------')
print('Author Name :', name)
print('Model :', model)
print('Summary :', summary)
print('---------------------------')


# Setting
pos_dir = "/data/train/positive/"
neg_dir = "/data/train/negative/"

features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)

norm = 'new' # 'norm' should be 'ws' or 'new'
do_resample = True

do_shuffle = True


# ------------------------------------------ Modify here ------------------------------------------------ #


# Data Load
X_train, y_train = train_data_loader(pos_dir, neg_dir, norm, do_resample, do_shuffle, features, target_voxel)


# ---------------------- Model parameters definition ----------------------- #
xgb_params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

log_params = {
    'penalty': ['l1', 'l2'],
    'tol': [1e-3, 1e-4, 1e-5],
    'C': [0.5, 1, 2],
}

sgd_params = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [1e-3, 1e-4, 1e-5],
    'max_iter': [1e+3, 2e+3, 4e+3]
}

svc_params = {
    'penalty': ['l1', 'l2'],
    'tol': [1e-3, 1e-4, 1e-5],
    'C': [0.5, 1, 2],
}

# ----------------------- Number of Combinations -------------------------- #
xgb_comb = np.prod([ len(val) for val in xgb_params.values() ])
log_comb = np.prod([ len(val) for val in log_params.values() ])
sgd_comb = np.prod([ len(val) for val in sgd_params.values() ])
svc_comb = np.prod([ len(val) for val in svc_params.values() ])
combos = [xgb_comb, log_comb, sgd_comb, svc_comb]

# scorer
scorer = make_scorer(fbeta_score, beta=0.5)


# --------------------------- Model definitions ---------------------------- #

xgb_base = xgb.XGBClassifier(learning_rate=0.01, n_estimators=600, objective='binary:logistic', silent=True, nthread=1)
log_base = LogisticRegression()
sgd_base = SGDClassifier()
svc_base = LinearSVC(dual=False)
bases = [xgb.XGBClassifier, LogisticRegression, SGDClassifier, LinearSVC]

xgb_model = RandomizedSearchCV(xgb_base, xgb_params, cv=5, scoring=scorer, verbose=3, n_jobs=-1)
log_model = RandomizedSearchCV(log_base, log_params, cv=5, scoring=scorer, verbose=3, n_jobs=-1)
sgd_model = RandomizedSearchCV(sgd_base, sgd_params, cv=5, scoring=scorer, verbose=3, n_jobs=-1)
svc_model = RandomizedSearchCV(svc_base, svc_params, cv=5, scoring=scorer, verbose=3, n_jobs=-1)
models = [xgb_model, log_model, sgd_model, svc_model]

# -------------------------- 1st Model fitting ----------------------------- #

for model in models:
    model.fit(X_train, y_train)


# ---------------------------- Ensemble ------------------------------------ #

estimators = []
for comb, model, base in zip(combos, models, bases):
    n = comb // 10 + 2  # use top 10% of the models

    df = pd.DataFrame(model.cv_results_)
    score = df['mean_train_score']
    score.index = df['params']
    score = score.sort_values()

    params = list(score.index)[-n:]

    for i, p in enumerate(params):
        estimators.append((base.__name__ + '_' + str(i), base(**p)))


final_model = VotingClassifier(estimators=estimators, voting="hard")


# ---------------------------- Model re-fitting ---------------------------- #


final_model.fit(X_train, y_train)
print(final_model)



# ----------------------------- Save model --------------------------------- #

pickle.dump(final_model, open('/data/model/final_model.pickle.dat', 'wb'))
