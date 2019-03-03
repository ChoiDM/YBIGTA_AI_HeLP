from utils.data_loader import train_data_loader
from utils.inference_tools import pred_to_binary, making_result
from utils.model_stacking import *
from utils.model_ml import *

import pickle
import os

import warnings
warnings.filterwarnings('ignore')



#------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- Training Settings ------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#

# Data Directory
path = "/data"
pos_dir = path+"/train/positive/"
neg_dir = path+"/train/negative/"



# Setting
random_state = 1213
deep = False
BETA = 0.75            # for Hyperparameter Tuning (Use F 0.75 Score)
BETA2 = 0.5            # for Stacking
cv=5                   # Cross validation

threshold = 'auto'     # Predict Top 'per_of_zero' percent to 1.
norm = 'lesion_based'  # 'lesion_based' or 'ws' (white-stripe)
per_of_zero = 45
do_minmax = True       # Min-max normalization for extracted features

include_model = [1,4,10,11,12]
include_model2 = [1,2,3,4]
include_model3 = []


# Data Loader
features = ['firstorder', 'shape']      # Which radiomics features to use
target_voxel = (0.65, 0.65, 3)          # Target resampling voxel size
do_resample = True
do_shuffle = True                       # Only do shuffle when training.

X_train, y_train = train_data_loader(pos_dir, neg_dir, norm, do_resample, do_shuffle, do_minmax, features, target_voxel, path=path)



#------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------- Fit ML model with training data -----------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#


# Model (1)
print("model1")
m1_params1 = {'subsample': [0.6], 'colsample_bytree': [0.6], 'min_child_weight': [0.5], 'probability': [True], 
              'gamma': [3.0], 'n_estimators': [300], 'learning_rate': [0.01], 'max_depth': [7], 'random_state' : [1213]}
model1 = ml_xgb(X_train, y_train, cv=cv, beta=BETA, params=None, random_state=random_state)


# Model (2)
print("\nmodel2")
m2_params1 = {'probability': [True], 'degree': [2], 'C': [0.001], 'gamma': [0.001]}
model2 = ml_svm(X_train, y_train, cv=cv, beta=BETA, params=m2_params1)


# Model (3)
print("\nmodel3")
model3 = ml_logistic(X_train, y_train, cv=cv, beta=BETA)


# Model (4)
print("\nmodel4")
m4_params1 = {'n_estimators': [500], 'min_samples_leaf': [50], 'max_depth': [15], 'random_state' : [1213]}
model4 = ml_rf(X_train, y_train, cv=cv, beta=BETA, params=None, random_state=random_state)


# Model (5)
print("\nmodel5")
m5_params1 = {'penalty': ['l1'], 'C': [1], 'max_iter': [900]}
model5 = ml_lasso(X_train, y_train, cv=cv, beta=BETA, params=m5_params1)


# Model (6)
print("\nmodel6")
m6_params1 =  {'alpha': [10], 'max_iter': [None]}
model6 = ml_ridge(X_train, y_train, cv=cv, beta=BETA, params=m6_params1)


# Model (7)
print("\nmodel7")
m7_params1 =  {'penalty': ['elasticnet'], 'loss': ['log'], 'alpha': [100], 'l1_ratio': [0.5], 'max_iter': [1400]}
model7 = ml_elasticNet(X_train, y_train, cv=cv, beta=BETA, params=m7_params1)


# Model (8)
print("\nmodel8")
m8_params1 =  {'n_nonzero_coefs': [70]}
model8 = ml_lars(X_train, y_train, cv=cv, beta=BETA, params=m8_params1)


# Model (9)
print("\nmodel9")
m9_params1 =  {'alpha': [0.1], 'max_iter': [800]}
model9 = ml_larsLasso(X_train, y_train, cv=cv, beta=BETA, params=m9_params1)


# Model (10)
print("\nmodel10")
m10_params1 =  {'n_estimators': [50], 'max_depth': [3], 'random_state' : [1213]}
model10 = ml_extraTrees(X_train, y_train, cv=cv, beta=BETA, params=None, random_state=random_state)


# Model (11)
print("\nmodel11")
model11 = ml_adaboost(X_train, y_train, cv=cv, beta=BETA, params=None, random_state=random_state)


# Model (12)
print("\nmodel12")
model12 = ml_lightgbm(X_train, y_train, cv=cv, beta=BETA, params=None, random_state=random_state)
#------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------
# Save ML Models to disk

pickle.dump(model1, open(path+'/model/model1.pickle.dat', 'wb'))
pickle.dump(model2, open(path+'/model/model2.pickle.dat', 'wb'))
pickle.dump(model3, open(path+'/model/model3.pickle.dat', 'wb'))
pickle.dump(model4, open(path+'/model/model4.pickle.dat', 'wb'))
pickle.dump(model5, open(path+'/model/model5.pickle.dat', 'wb'))
pickle.dump(model6, open(path+'/model/model6.pickle.dat', 'wb'))
pickle.dump(model7, open(path+'/model/model7.pickle.dat', 'wb'))
pickle.dump(model8, open(path+'/model/model8.pickle.dat', 'wb'))
pickle.dump(model9, open(path+'/model/model9.pickle.dat', 'wb'))
pickle.dump(model10, open(path+'/model/model10.pickle.dat', 'wb'))
pickle.dump(model11, open(path+'/model/model11.pickle.dat', 'wb'))
pickle.dump(model12, open(path+'/model/model12.pickle.dat', 'wb'))

#------------------------------------------------------------------------------------------------------------------------
# Fit stacking model

# Layer1
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]
S_train = stacking(models, X_train, include_model)

meta_xgb = stacking_xgb(S_train, y_train, cv=cv, beta=BETA2)
meta_logistic = stacking_logistic(S_train, y_train, cv=cv, beta=BETA2)
meta_NN = stacking_NN(S_train, y_train, deep=deep)
meta_weight = stacking_weight(S_train, y_train)

y_pred_lst = []
y_pred_binary_lst =[]
y_pred_lst2 = []
y_pred_binary_lst2 =[]

for meta in [meta_xgb, meta_logistic, meta_NN, meta_weight] :
    pred = meta.predict_proba(S_train)[:, 1]
    y_pred_lst.append(pred)
    y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold, per_of_zero=per_of_zero))

    
# Print result
print("\n")
print(making_result(S_train, y_pred_lst, y_pred_binary_lst, y_pred_lst2, y_pred_binary_lst2, include_model, include_model2, include_model3, y_train))\

#------------------------------------------------------------------------------------------------------------------------
# Save stacking model 1
pickle.dump(meta_xgb, open(os.path.join(path, '/model/meta_xgb.pickle.dat', 'wb')))
pickle.dump(meta_logistic, open(os.path.join(path, '/model/meta_logistic.pickle.dat', 'wb')))

meta_NN.model.save_weights(os.path.join(path, '/model/meta_NN.h5'))
with open(os.path.join(path, '/model/meta_NN.json', 'w')) as f :
    f.write(meta_NN.model.to_json())
    
meta_weight.model.save_weights(os.path.join(path,'/model/meta_weight.h5'))
with open(os.path.join(path, '/model/meta_weight.json', 'w')) as f :
    f.write(meta_weight.model.to_json())
#------------------------------------------------------------------------------------------------------------------------



