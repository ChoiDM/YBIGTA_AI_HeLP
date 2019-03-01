from utils.data_loader import train_data_loader, test_data_loader, data_generator
from utils.inference_tools import pred_to_binary, export_csv, making_result,  error_check
from utils.model_stacking import *
from utils.cube_tools import *
from utils.model_ml import *
import vecstack
from glob import glob

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, SGDClassifier, Lars, LassoLars
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Conv3D, Flatten, pooling
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

path = "/data"
pos_dir = path+"/train/positive/"
neg_dir = path+"/train/negative/"
save_dir = path+"/model/"
test_dir = path+'/test/'


# Setting
# Set your params here!!!
threshold = "auto"
norm = 'new'
mode = 'train'

## MLP
num_units=256
hidden_layers=3
epochs1=30
loss="cross_entropy_loss"

## CNN
epochs2 = 15
batch_size = 4
cube_shape = (32, 32, 16)


# Print Information
name = 'KHW2_DL'
model = 'MLP & CNN'
summary1 = 'Hyperparams with MLP & CNN  :  threshold={} + norm={}'.format(threshold, norm)
summary2 = "--- MLP : units={} + hidden_layers={} + epochs={} + loss={}\n --- CNN : epochs={} + batch_size={} + cube_shape={}".format(num_units, hidden_layers, epochs1, loss, epochs2, batch_size, cube_shape)

print('Author Name :', name)
print('Model :', model)
print('Summary :', summary1)
print('Summary2 \n', summary2)


# Data Load
print("\n---------- Data Load ----------")
features = ['firstorder', 'shape']
target_voxel = (0.65, 0.65, 3)
do_resample = True
do_shuffle = True
do_minmax = True

X_train, y_train = train_data_loader(pos_dir, neg_dir, norm, do_resample, do_shuffle, do_minmax, features, target_voxel, path=path)


####################################################################z#####################################################
#########################################################################################################################
#### Modify here ####

#------------------------------------------------------------------------------------------------------------------------
# Fit model with training data
print("\n---------- Start Train ----------")

#########
## model1
print("model1")
MLP = dl_mlp(X_train, y_train, num_units=num_units, hidden_layers=hidden_layers, epochs=epochs1, loss=loss)

#########
## model2
print("\nmodel2")

data_dir = sorted(glob(os.path.join(path, mode, '*', '*')))
data_dir, error_patient = error_check(data_dir)
data_gen = data_generator(batch_size, mode, data_dir, cube_shape, norm, target_voxel)

CNN = dl_cnn(data_gen, cube_shape=cube_shape, batch_size=batch_size, epochs=epochs2)
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Save model
print("\n---------- Save Model ----------")

MLP.model.save_weights(path+'/model/MLP.h5')
with open(path+'/model/MLP.json', 'w') as f :
    f.write(MLP.model.to_json())
    
CNN.model.save_weights(path+'/model/CNN.h5')
with open(path+'/model/CNN.json', 'w') as f :
    f.write(CNN.model.to_json())
#------------------------------------------------------------------------------------------------------------------------

print("\n---------- train.py finished ----------")
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")



