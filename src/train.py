from utils.data_loader import train_data_loader

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import pickle
import datetime

import warnings
warnings.filterwarnings('ignore')


# Print Current Time
time = str(datetime.datetime.now()).split()[1].split('.')[0]
print("Start:", time)


# Print Information
name = 'KHW'
model = 'SVM'
summary = 'No hyper-parameter tuning. Basic Model'

print('Author Name :', name)
print('Model :', model)
print('Summary :', summary)
print("\n")


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
X_train, y_train = train_data_loader(pos_dir, neg_dir, do_n4, do_ws, do_resample, do_shuffle, save_to_disk, return_patient_num)


#########################################################################################################################
#########################################################################################################################
#### Modify here ####


# Fit Model with Training Data
print("\nStart Train...")
model = SVC()
model.fit(X_train, y_train)


# Save model to file
pickle.dump(model, open('/data/model/model.pickle.dat', 'wb'))
