from utils.data_loader import train_data_loader, test_data_loader
from utils.inference_tools import making_df

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


# Print Current Time
time = str(datetime.datetime.now()).split()[1].split('.')[0]
print("---------- Start ----------")
print("Start:", time, "\n")


# Print Information
name = 'KHW'
model = 'Stacking sklearn model'
summary = 'HyperParams tuning with 9 sklearn models'

print('Author Name :', name)
print('Model :', model)
print('Summary :', summary)


# Setting
pos_dir = "/data/train/positive/"
neg_dir = "/data/train/negative/"

do_n4 = False
do_ws = True
do_resample = True

do_shuffle = True
save_to_disk = False
return_patient_num = False


# Data Load
print("\n---------- Data Load ----------")
X_train, y_train = train_data_loader(pos_dir, neg_dir, do_n4, do_ws, do_resample, do_shuffle, save_to_disk, return_patient_num)


#########################################################################################################################
#########################################################################################################################
#### Modify here ####


# Fit ML model with training data
print("\n---------- Start ML Train ----------")

#########
## scorer
def new_scorer(y_true, y_pred, threshold=0.5) :
    result = []

    for pred in list(y_pred) :
        if pred >= threshold :
            result.append(1)
        else :
            result.append(0)
            
    return fbeta_score(y_true, np.array(result), beta=0.5)

scorer = make_scorer(fbeta_score, beta=0.5)

#########
## model1
print("model1")
model1 = XGBClassifier()

m1_params1 = {
    'max_depth' : [5,6,7,8],
    'min_child_weight' : [0.5, 1, 5, 10, 15, 20],
    'gamma' : [1.5, 2, 2.5, 3.0, 5],
    'subsample' : [0.5, 0.6, 0.8, 1.0],
    'colsample_bytree' : [0.5, 0.6, 0.8, 1.0],
    'probability' : [True],
    'learning_rate' : [0.01, 0.05, 0.1],
    'n_estimators' : [300, 500, 700]
}

m1_grid_1 = GridSearchCV(model1, param_grid=m1_params1, scoring=scorer, cv=2, verbose=0, n_jobs=-1)
m1_grid_1.fit(X_train, y_train)

best_model1 = m1_grid_1.best_estimator_

print("Best Score : {}".format(m1_grid_1.best_score_))
print("Best Params : {}".format(m1_grid_1.best_params_))

#########
## model2
print("\nmodel2")
model2 = SVC()

m2_params1 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'gamma' : [0.001, 0.01, 0.1, 1, 2, 5, 10, 20],
    'degree' : [2,3,4],
    'probability' : [True]
}

m2_grid_1 = GridSearchCV(model2, param_grid=m2_params1, scoring=scorer, cv=2, verbose=0, n_jobs=-1)
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
    'max_iter' : [n for n in range(100,1101, 200)],
}

m3_grid_1 = GridSearchCV(model3, param_grid=m3_params1, scoring=scorer, cv=2, verbose=0, n_jobs=-1)
m3_grid_1.fit(X_train, y_train)

best_model3 = m3_grid_1.best_estimator_

print("Best Score : {}".format(m3_grid_1.best_score_))
print("Best Params : {}".format(m3_grid_1.best_params_))

#########
## model4
print("\nmodel4")
model4 = RandomForestClassifier()

m4_params1 = {
    'max_depth' : [6, 8, 10, 15, 20, 30, 40, 50],
    'min_samples_leaf': [1, 2, 3, 4, 5,10, 20, 50],
    'n_estimators' : [100, 300, 500]
}

m4_grid_1 = GridSearchCV(model4, param_grid=m4_params1, scoring=scorer, cv=2, verbose=0, n_jobs=-1)
m4_grid_1.fit(X_train, y_train)

best_model4 = m4_grid_1.best_estimator_

print("Best Score : {}".format(m4_grid_1.best_score_))
print("Best Params : {}".format(m4_grid_1.best_params_))

#########
## model5
print("\nmodel5")
model5 = LogisticRegression()

m5_params1 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter' : [n for n in range(100,1101, 200)],
    'penalty' : ["l1"]
}

m5_grid_1 = GridSearchCV(model5, param_grid=m5_params1, scoring=scorer, cv=2, verbose=0, n_jobs=-1)
m5_grid_1.fit(X_train, y_train)

best_model5 = m5_grid_1.best_estimator_

print("Best Score : {}".format(m5_grid_1.best_score_))
print("Best Params : {}".format(m5_grid_1.best_params_))

#########
## model6
print("\nmodel6")
model6 = RidgeClassifier()

m6_params1 = {
    'alpha': [0.1, 1, 2, 5, 10, 20, 50, 100],
    'max_iter' : [None]+[n for n in range(100,1101, 200)]
}

m6_grid_1 = GridSearchCV(model6, param_grid=m6_params1, scoring=scorer, cv=2, verbose=0, n_jobs=-1)
m6_grid_1.fit(X_train, y_train)

best_model6 = m6_grid_1.best_estimator_

print("Best Score : {}".format(m6_grid_1.best_score_))
print("Best Params : {}".format(m6_grid_1.best_params_))

#########
## model7
print("\nmodel7")
model7 = SGDClassifier()

m7_params1 = {
    'alpha': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 50, 100],
    'l1_ratio':[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 
    'max_iter' : [None]+[n for n in range(800, 1601, 200)],
    'penalty' : ["elasticnet"],
    'loss' : ["log"]
}

m7_grid_1 = GridSearchCV(model7, param_grid=m7_params1, scoring=scorer, cv=2, verbose=0, n_jobs=-1)
m7_grid_1.fit(X_train, y_train)

best_model7 = m7_grid_1.best_estimator_

print("Best Score : {}".format(m7_grid_1.best_score_))
print("Best Params : {}".format(m7_grid_1.best_params_))

#########
## model8
print("\nmodel8")
model8 = Lars()

m8_params1 = {
    'n_nonzero_coefs': [n for n in range(30, 150, 20)]
}

max_score=0
m8_best_t = 0
best_model8 = ""
m8_best_grid_1 = ""

for t in [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.45, 0.4, 0.45, 0.5, 0.6] :
    scorer2 = make_scorer(new_scorer, threshold=t)
    m8_grid_1 = GridSearchCV(model8, param_grid=m8_params1, scoring=scorer2, cv=2, verbose=0, n_jobs=-1)
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

m9_params1 = {
    'alpha': [0.1, 1, 2, 5, 10, 20, 50, 100],
    'max_iter' : [n for n in range(800, 1601, 200)]
}

max_score=0
m9_best_t = 0
best_model9 = ""
m9_best_grid_1 = ""
for t in [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.45, 0.4, 0.45, 0.5, 0.6] :
    scorer2 = make_scorer(new_scorer, threshold=t)
    m9_grid_1 = GridSearchCV(model9, param_grid=m9_params1, scoring=scorer2, cv=2, verbose=0, n_jobs=-1)
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

m10_params1 = {
    'max_depth' : [None, 3, 5, 7, 9],
    'n_estimators' : [10, 50, 100, 300, 500]
}

m10_grid_1 = GridSearchCV(model10, param_grid=m10_params1, scoring=scorer, cv=2, verbose=0, n_jobs=-1)
m10_grid_1.fit(X_train, y_train)

best_model10 = m10_grid_1.best_estimator_

print("Best Score : {}".format(m10_grid_1.best_score_))
print("Best Params : {}".format(m10_grid_1.best_params_))


# Save ML model
print("\n---------- Save ML Model ----------")
pickle.dump(best_model1, open('../data/model/model1.pickle.dat', 'wb'))
pickle.dump(best_model2, open('../data/model/model2.pickle.dat', 'wb'))
pickle.dump(best_model3, open('../data/model/model3.pickle.dat', 'wb'))
pickle.dump(best_model4, open('../data/model/model4.pickle.dat', 'wb'))
pickle.dump(best_model5, open('../data/model/model5.pickle.dat', 'wb'))
pickle.dump(best_model6, open('../data/model/model6.pickle.dat', 'wb'))
pickle.dump(best_model7, open('../data/model/model7.pickle.dat', 'wb'))
pickle.dump(best_model8, open('../data/model/model8.pickle.dat', 'wb'))
pickle.dump(best_model9, open('../data/model/model9.pickle.dat', 'wb'))
pickle.dump(best_model10, open('../data/model/model10.pickle.dat', 'wb'))

# Model Stacking
print("\n---------- Model Stacking ----------")
def stacking(models, data) : 
    result = []
    
    for idx, model in enumerate(models) :
        if idx+1 in [2,6,7,8,9] :
            continue
        if idx+1 in ["None"] :
            result.append(model.predict(data))
        else :
            result.append(model.predict_proba(data)[:,1])
        print("model", idx+1, "is stacked")
        
    return np.array(result).T

models = [best_model1, best_model2, best_model3, best_model4, best_model5, best_model6, best_model7, best_model8, best_model9, best_model10]
S_train = stacking(models, X_train)
print(S_train)

# Fit stacking model
print("\n---------- Start Staking Train ----------")
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

def stack_fn(num_models=len(S_train[0])):
    model = Sequential()
    model.add(Dense(16, input_dim=num_models, activation='relu'))
    model.add(Dense(16, input_dim=16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

meta_model = KerasClassifier(build_fn=stack_fn)
meta_model.fit(S_train, y_train, epochs=30)
print(predict_train(S_train, meta_model.predict_proba(S_train), y_train))


# Save stacking model
print("\n---------- Save Staking Model ----------")
meta_model.model.save_weights('/data/model/model_weights.h5')

with open('/data/model/model_architecture.json', 'w') as f :
    f.write(meta_model.model.to_json())





