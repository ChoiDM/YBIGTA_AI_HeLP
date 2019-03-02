from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, SGDClassifier, Lars, LassoLars
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Conv3D, Flatten, pooling, Activation
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD, Adam

import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

def dl_cnn(data_gen, cube_shape=(32,32,16), batch_size=4, epochs=20, seq=None) :
    input_shape = cube_shape + (2,)
    
    if seq is None:
        steps_per_epoch =255//batch_size
    else:
        steps_per_epoch =255//batch_size*len(seq)
    
    model = Sequential()
    model.add(Conv3D(32, (3,3,3), activation=None, input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(32, (3,3,3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(pooling.MaxPooling3D(pool_size=(2,2,2)))
    
    model.add(Conv3D(64, (3,3,3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(64, (3,3,3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(pooling.MaxPooling3D(pool_size=(2,2,2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit_generator(data_gen, steps_per_epoch, epochs, shuffle=False)  
    return model

def dl_mlp(X_train, y_train, batch_size, optimizer='adam', lr='0.01', num_units=256, hidden_layers=3, epochs=30, loss="BCE") :
    
    def stack_fn(num_models=X_train.shape[1], num_units=num_units, hidden_layers=hidden_layers, loss=loss):
        
        # Build Model
        model = Sequential()
        
        for _ in range(hidden_layers) :
            model.add(Dense(num_units, input_dim=num_models, activation=None))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.3))
        
        model.add(Dense(16, input_dim=num_units, activation='relu'))
        
        
        # Loss Function
        if loss == 'BCE':
            model.add(Dense(2, activation='sigmoid'))
            loss_func = 'binary_crossentropy'
        
        elif loss == 'CE':
            model.add(Dense(2, activation='softmax'))
            loss_func = 'categorical_crossentropy'
        
        elif loss == 'focal':
            model.add(Dense(2, activation='softmax'))
            loss_func = focal_loss()
            
        
        # Optimizer
        if optimizer == 'adam':
            opt = Adam(lr=lr)
        
        elif optimizer == 'sgd':
            opt = SGD(lr=lr, momentum=0.9, decay=1e-6)
            
            
        # Compile
        model.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
            
        return model
    
    MLP_model = KerasClassifier(build_fn=stack_fn)    
    MLP_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    
    return MLP_model

def ml_xgb(X_train, y_train, cv=5, beta=0.75, params = None, random_state=1213) :
    model = XGBClassifier()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'max_depth' : [5,6,7,8],
            'min_child_weight' : [0.5, 1, 5, 10, 15, 20],
            'gamma' : [1.5, 2, 2.5, 3.0, 5],
            'subsample' : [0.5, 0.6, 0.8, 1.0],
            'colsample_bytree' : [0.5, 0.6, 0.8, 1.0],
            'probability' : [True],
            'learning_rate' : [0.01, 0.05, 0.1],
            'n_estimators' : [300, 500, 700], 
            'random_state' : [random_state]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def ml_svm(X_train, y_train, cv=5, beta=0.75, params = None) :
    model = SVC()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100], 
            'gamma' : [0.001, 0.01, 0.1, 1, 2, 5, 10, 20],
            'degree' : [2,3,4],
            'probability' : [True]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def ml_logistic(X_train, y_train, cv=5, beta=0.75, params = None) :
    model = LogisticRegression()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter' : [n for n in range(100,1101, 200)]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def ml_rf(X_train, y_train, cv=5, beta=0.75, params = None, random_state=1213) :
    model = RandomForestClassifier()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'max_depth' : [6, 8, 10, 15, 20, 30, 40, 50],
            'min_samples_leaf': [1, 2, 3, 4, 5,10, 20, 50],
            'n_estimators' : [100, 300, 500], 
            'random_state' : [random_state]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def ml_lasso(X_train, y_train, cv=5, beta=0.75, params = None) :
    model = LogisticRegression()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter' : [n for n in range(100,1101, 200)],
            'penalty' : ["l1"]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def ml_ridge(X_train, y_train, cv=5, beta=0.75, params = None) :
    model = RidgeClassifier()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'alpha': [0.1, 1, 2, 5, 10, 20, 50, 100],
            'max_iter' : [None]+[n for n in range(100,1101, 200)]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def ml_elasticNet(X_train, y_train, cv=5, beta=0.75, params = None) :
    model = SGDClassifier()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'alpha': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 50, 100],
            'l1_ratio':[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 
            'max_iter' : [None]+[n for n in range(800, 1601, 200)],
            'penalty' : ["elasticnet"],
            'loss' : ["log"]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def ml_lars(X_train, y_train, cv=5, beta=0.75, params = None) :
    model = Lars()
    
    if not params :
        params = {
            'n_nonzero_coefs': [n for n in range(30, 150, 20)]
        }

    max_score=0
    best_t = 0
    best_model = ""
    best_grid = ""

    for t in [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.45, 0.4, 0.45, 0.5, 0.6] :
        scorer = make_scorer(new_scorer, threshold=t, beta=beta)
        model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
        model_grid.fit(X_train, y_train)

        if max_score < model_grid.best_score_ :
            best_model = model_grid.best_estimator_
            best_t = t
            best_grid = model_grid

    model_grid = best_grid
    best_model = best_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))     
    print("Threshold :", best_t)
    print("Best Params : {}".format(model_grid.best_params_))
    
    
    
def ml_larsLasso(X_train, y_train, cv=5, beta=0.75, params = None) :
    model = LassoLars()
    
    if not params :
        params = {
            'alpha': [0.1, 1, 2, 5, 10, 20, 50, 100],
            'max_iter' : [n for n in range(800, 1601, 200)]
        }

    max_score=0
    best_t = 0
    best_model = ""
    best_grid = ""

    for t in [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.45, 0.4, 0.45, 0.5, 0.6] :
        scorer = make_scorer(new_scorer, threshold=t, beta=beta)
        model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
        model_grid.fit(X_train, y_train)

        if max_score < model_grid.best_score_ :
            best_model = model_grid.best_estimator_
            best_t = t
            best_grid = model_grid

    model_grid = best_grid
    best_model = best_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))     
    print("Threshold :", best_t)
    print("Best Params : {}".format(model_grid.best_params_))
    
def ml_extraTrees(X_train, y_train, cv=5, beta=0.75, params = None, random_state=1213) :
    model = ExtraTreesClassifier()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'max_depth' : [None, 3, 5, 7, 9],
            'n_estimators' : [10, 50, 100, 300, 500], 
            'random_state' : [random_state]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def ml_adaboost(X_train, y_train, cv=5, beta=0.75, params = None, random_state=1213) :
    model = AdaBoostClassifier()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'n_estimators' : [100, 300, 500],
            'learning_rate' : [0.01, 0.05, 0.1],
            'algorithm' :['SAMME.R'], 
            'random_state' : [random_state]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def ml_lightgbm(X_train, y_train, cv=5, beta=0.75, params = None, random_state=1213) :
    model = LGBMClassifier()
    scorer = make_scorer(fbeta_score, beta=beta)
    
    if not params :
        params = {
            'min_child_weight' : [0.5, 1, 2, 5],
            'colsample_bytree' : [0.6, 0.8, 1.0],
            'subsample' : [0.6, 0.8, 1.0],
            'learning_rate' : [0.05, 0.1],
            'n_estimators' : [100, 300],
            'reg_alpha' : [0.0, 1.0, 2.0, 5.0], 
            'reg_lambda' : [0.0, 1.0, 2.0, 5.0], 
            'random_state' : [random_state]
        }
    
    model_grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=cv, verbose=0, n_jobs=-1)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    print("Best Score : {}".format(model_grid.best_score_))
    print("Best Params : {}".format(model_grid.best_params_))
    
    return best_model

def new_scorer(y_true, y_pred, threshold=0.5, beta=0.75) :
        result = []

        for pred in list(y_pred) :
            if pred >= threshold :
                result.append(1)
            else :
                result.append(0)

        return fbeta_score(y_true, np.array(result), beta=beta)
