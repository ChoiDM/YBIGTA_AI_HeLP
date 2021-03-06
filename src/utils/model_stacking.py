from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

def get_stacking_base_model(models, models_num) :
    result = []
    for idx in models_num :
        result.append(models[idx-1])
        
    return result

def stacking(models, data, include, predict_binary=[None]) : 
    result = []
    
    for idx, model in enumerate(models) :
        if idx+1 not in include :
            continue
                
        if idx+1 in predict_binary :
            result.append(model.predict(data))
        else :
            result.append(model.predict_proba(data)[:,1])
            print("model", idx+1, "is stacked")
 
    print("\n")
    return np.array(result).T


def stacking_xgb(S_train, y_train, stacking_params=None, cv=5, beta=0.5, random_state=1213) :
    stacking_model = XGBClassifier()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not stacking_params :
        stacking_params = {
            'max_depth' : [2,3],
            'min_child_weight' : [0.5, 1, 5, 10],
            'gamma' : [1.5, 2, 2.5, 3.0, 5],
            'subsample' : [0.5, 0.6, 0.8, 1.0],
            'colsample_bytree' : [0.5, 0.6, 0.8, 1.0],
            'probability' : [True],
            'learning_rate' : [0.01, 0.05, 0.1],
            'n_estimators' : [100, 200, 300], 
            'random_state' : [random_state]
        }
    
    stacking_grid = GridSearchCV(stacking_model, param_grid=stacking_params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    stacking_grid.fit(S_train, y_train)

    meta_model = stacking_grid.best_estimator_
    print("Best Score : {}".format(stacking_grid.best_score_))
    print("Best Params : {}".format(stacking_grid.best_params_))
    
    return meta_model
    
    
def stacking_logistic(S_train, y_train, stacking_params=None, cv=5, beta=0.5) :
    stacking_model = LogisticRegression()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not stacking_params :
        stacking_params =  {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'max_iter' : [n for n in range(100, 1101, 200)],
        }
    
    stacking_grid = GridSearchCV(stacking_model, param_grid=stacking_params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    stacking_grid.fit(S_train, y_train)

    meta_model = stacking_grid.best_estimator_
    print("Best Score : {}".format(stacking_grid.best_score_))
    print("Best Params : {}".format(stacking_grid.best_params_))
    
    return meta_model
    
def stacking_weight(S_train, y_train, cv=5, epochs=20) :
    def stack_fn(num_models=len(S_train[0])):
        model = Sequential()
        model.add(Dense(2, input_dim=num_models, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    meta_model = KerasClassifier(build_fn=stack_fn)
    meta_model.fit(S_train, y_train, epochs=epochs)
    return meta_model
    
def stacking_NN(S_train, y_train, cv=5, epochs=20, deep = False) :
    def stack_fn(num_models=len(S_train[0])):
        model = Sequential()
        model.add(Dense(16, input_dim=num_models, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def stack_fn2(num_models=len(S_train[0])):
        model = Sequential()
        model.add(Dense(8, input_dim=num_models, activation='relu'))
        model.add(Dense(8, input_dim=8, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    if deep :
        meta_model = KerasClassifier(build_fn=stack_fn2)
    else :
        meta_model = KerasClassifier(build_fn=stack_fn)
        
    meta_model.fit(S_train, y_train, epochs=epochs)
    return meta_model


