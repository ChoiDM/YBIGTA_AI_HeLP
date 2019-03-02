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
#from imgaug import augmenters as iaa

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

## Data Augmentation for CNN
"""
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.Flipud(0.2), # vertically flip 20% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
    iaa.Afine(rotate=(-10, 10))
], random_order = True)
"""

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
data_gen = data_generator(batch_size, mode, data_dir, cube_shape, norm, target_voxel, seq=None)

CNN = dl_cnn(data_gen, cube_shape=cube_shape, batch_size=batch_size, epochs=epochs2 , seq=None)
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



