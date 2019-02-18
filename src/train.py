from utils.data_loader import train_data_loader
from sklearn.metrics import fbeta_score, make_scorer
import xgboost as xgb
import pickle
import datetime

import warnings
warnings.filterwarnings('ignore')

# Print Current Time
time = str(datetime.datetime.now()).split()[1].split('.')[0]
print("Start:", time)


# Print Information
name = 'master-branch'
model = 'XGBoost'
summary = 'Parameter from Grid-Search (subsample=1.0, gamma=2, colsample_bytree=0.8, max_depth=6, min_child_weight=10)'

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


#########################################################################################################################
#########################################################################################################################
#### Modify here ####


# Data Load
X_train, y_train = train_data_loader(pos_dir, neg_dir, norm, do_resample, do_shuffle, features, target_voxel)


# Fit Model with Training Data
xgbClassifier = xgb.XGBClassifier(subsample=1.0, gamma=2, colsample_bytree=0.8, max_depth=6, min_child_weight=10)
xgbClassifier.fit(X_train, y_train)


# Use F0.5 score if necessary
# f_score = make_scorer(fbeta_score, beta = 0.5)


# Save model to file
pickle.dump(xgbClassifier, open('/data/model/xgb.pickle.dat', 'wb'))
