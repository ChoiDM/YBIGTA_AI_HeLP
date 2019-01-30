from utils.data_loader import train_data_loader
import xgboost as xgb
import pickle

import warnings
warnings.filterwarnings('ignore')

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
xgbClassifier = xgb.XGBClassifier()
xgbClassifier.fit(X_train, y_train)


# Save model to file
pickle.dump(xgbClassifier, open('xgb.pickle.dat', 'wb'))
