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
name = 'main-branch'
model = 'XGBoost'
summary = 'No hyper-parameter tuning. Basic Model'

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
X_train, y_train = train_data_loader(pos_dir, neg_dir, do_n4, do_ws, do_resample, do_shuffle, save_to_disk, return_patient_num,
                                     features, target_voxel)


# Fit Model with Training Data
xgbClassifier = xgb.XGBClassifier()
xgbClassifier.fit(X_train, y_train)


# Use F0.5 score if necessary
# f_score = make_scorer(fbeta_score, beta = 0.5)


# Save model to file
pickle.dump(xgbClassifier, open('/data/model/xgb.pickle.dat', 'wb'))
