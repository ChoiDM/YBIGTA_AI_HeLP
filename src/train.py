from utils.data_loader import train_data_loader, test_data_loader
from utils.inference_tools import pred_to_binary, export_csv, making_result
from utils.model_stacking import *
from utils.model_ml import *
import vecstack

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, SGDClassifier, Lars, LassoLars
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from keras.models import Sequential
from keras.layers import Dense, Dropout
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

path = "./notebook/data"
pos_dir = path+"/train/positive/"
neg_dir = path+"/train/negative/"
save_dir = path+"/model/"
test_dir = path+'/test/'


# Setting
# Set your params here!!!
threshold = "auto"
norm = 'new'
num_units=256 
hidden_layers=3
epochs=30
loss="cross_entropy_loss"
gamma = 2.0
alpha = 0.25


# Print Information
name = 'KHW2_MLP'
model = 'MLP'
summary1 = 'Hyperparams with MLP'
summary2 = "threshold={} + norm={} + num_units={} + hidden_layers={} + epochs={} + loss={} + gamma={} + alpha={}".format(threshold, norm, num_units, hidden_layers, epochs, loss, gamma, alpha)

print('Author Name :', name)
print('Model :', model)
print('Summary :', summary1)
print('Summary2 :', summary2)


# Data Load
print("\n---------- Data Load ----------")
features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)
do_resample = True
do_shuffle = True

X_train, y_train = train_data_loader(pos_dir, neg_dir, norm, do_resample, do_shuffle, features, target_voxel)
X_test, patient_num, error_patient = test_data_loader(test_dir, norm, do_resample, features, target_voxel)

np.save(save_dir+"X_train.npy", X_train)
np.save(save_dir+"y_train.npy", y_train)


####################################################################z#####################################################
#########################################################################################################################
#### Modify here ####

#------------------------------------------------------------------------------------------------------------------------
# Fit model with training data
print("\n---------- Start Train ----------")

#########
## model1
def MLP_layers(X_train, y_train, num_units=256, hidden_layers=3, epochs=30, loss="cross_entropy_loss", gamma=2.0, alpha=0.25) :
    
    def focal_loss(gamma=gamma, alpha=alpha) :
        def focal_loss_fixed(y_true, y_pred):
            eps = 1e-12
            y_pred=K.clip(y_pred, eps, 1.0-eps)
            
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha*K.pow(1.0-pt_1, gamma)*K.log(pt_1)) -K.sum((1-alpha)*K.pow(pt_0, gamma)*K.log(1.0-pt_0))
        return focal_loss_fixed

    def stack_fn(num_models=X_train.shape[1], num_units=num_units, hidden_layers=hidden_layers, loss=loss):
        model = Sequential()
        
        for _ in range(hidden_layers) :
            model.add(Dense(num_units, input_dim=num_models, activation='relu'))
            model.add(Dropout(0.5))
        
        model.add(Dense(32, input_dim=num_units, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        
        if loss == 'cross_entropy_loss' :
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif loss == 'focal_loss' :
            model.compile(loss=focal_loss, optimizer='adam', metrics=['accuracy'])
            
        return model
    
    MLP_model = KerasClassifier(build_fn=stack_fn)    
    MLP_model.fit(X_train, y_train, epochs=epochs)
    return MLP_model

MLP = MLP_layers(X_train, y_train, num_units=num_units, hidden_layers=hidden_layers, epochs=epochs, loss=loss, gamma=gamma, alpha=alpha)
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Save model
print("\n---------- Save Model ----------")

MLP.model.save_weights(path+'/model/MLP.h5')
with open(path+'/model/MLP.json', 'w') as f :
    f.write(MLP.model.to_json())
#------------------------------------------------------------------------------------------------------------------------

print("\n---------- train.py finished ----------")
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")



