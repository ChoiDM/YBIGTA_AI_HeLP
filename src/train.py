import numpy as np
from utils.data_loader import train_data_loader
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import datetime

import warnings
warnings.filterwarnings('ignore')

time = str(datetime.datetime.now()).split()[1].split('.')[0]
print("Start:", time)
print("Author: Semin")

# Setting
pos_dir = "/data/train/positive/"
neg_dir = "/data/train/negative/"

do_n4 = False
do_ws = True
do_resample = True

do_shuffle = True
save_to_disk = False
return_patient_num = False


#########################################################################################################################
#########################################################################################################################
#### Modify here ####


# Data Load
X_train, y_train = train_data_loader(pos_dir, neg_dir, do_n4, do_ws, do_resample, do_shuffle, save_to_disk, return_patient_num)


# Fit Model with Training Data
params = {
    'n_estimators': [10,50,100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 3, 5, 10],
    'max_features': ['sqrt', 'log2']
}
rf = RandomForestClassifier(n_jobs=-1)
model = GridSearchCV(rf, params, cv=5)
model.fit(X_train, y_train)

print("Best params:", model.best_params_)
print("Best estimator:\n", model.best_estimator_)

cvres = model.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

# Save model to file
pickle.dump(model, open('/data/model/model.pickle.dat', 'wb'))
