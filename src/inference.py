from utils.data_loader import test_data_loader
from utils.inference_tools import pred_to_binary, export_csv, making_result
from utils.model_stacking import *

import pickle
import os

import warnings
warnings.filterwarnings('ignore')


#------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- Inference Settings -----------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#
path = "/data"
test_dir = path+'/test/'


random_state = 1213
threshold = 'auto'     # Predict Top 'per_of_zero' percent to 1.
norm = 'lesion_based'  # 'lesion_based' or 'ws' (white-stripe)
per_of_zero = 45
do_minmax = True       # Min-max normalization for extracted features


include_model = [1,4,10,11,12]
include_model2 = [1,2,3,4]
include_model3 = []

final_idx = 3          # 1=XGB, 2=Logistic, 3=NN, 4=Weight


# Data Loader
features = ['firstorder', 'shape']      # Which radiomics features to use
target_voxel = (0.65, 0.65, 3)          # Target resampling voxel size
do_resample = True

X_test, patient_num, error_patient = test_data_loader(test_dir, norm, do_resample, do_minmax, features, target_voxel, path=path)


#------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- Load Trained model  -----------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#
model1 = pickle.load(open(path+'/model/model1.pickle.dat', 'rb'))
model2 = pickle.load(open(path+'/model/model2.pickle.dat', 'rb'))
model3 = pickle.load(open(path+'/model/model3.pickle.dat', 'rb'))
model4 = pickle.load(open(path+'/model/model4.pickle.dat', 'rb'))
model5 = pickle.load(open(path+'/model/model5.pickle.dat', 'rb'))
model6 = pickle.load(open(path+'/model/model6.pickle.dat', 'rb'))
model7 = pickle.load(open(path+'/model/model7.pickle.dat', 'rb'))
model8 = pickle.load(open(path+'/model/model8.pickle.dat', 'rb'))
model9 = pickle.load(open(path+'/model/model9.pickle.dat', 'rb'))
model10 = pickle.load(open(path+'/model/model10.pickle.dat', 'rb'))
model11 = pickle.load(open(path+'/model/model11.pickle.dat', 'rb'))
model12 = pickle.load(open(path+'/model/model12.pickle.dat', 'rb'))


#------------------------------------------------------------------------------------------------------------------------
# Load Stacking model 1
meta_xgb = pickle.load(open(os.path.join(path, '/model/meta_xgb.pickle.dat', 'rb')))
meta_logistic = pickle.load(open(os.path.join(path, '/model/meta_logistic.pickle.dat', 'rb')))

with open(os.path.join(path, '/model/meta_NN.json', 'r')) as f :
    meta_NN = model_from_json(f.read())
meta_NN.model.load_weights(os.path.join(path, '/model/meta_NN.h5'))

with open(os.path.join(path, '/model/meta_weight.json', 'r')) as f :
    meta_weight = model_from_json(f.read())
meta_weight.model.load_weights(os.path.join(path, '/model/meta_weight.h5'))


#------------------------------------------------------------------------------------------------------------------------
# Stacking model
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]
models2 = [meta_xgb, meta_logistic, meta_NN, meta_weight]
models3 = []

# Layer1
S_test = stacking(models, X_test, include_model)

y_pred_lst = []
y_pred_binary_lst =[]
y_pred_lst2 = []
y_pred_binary_lst2 =[]

for meta in models2 :
    pred = meta.predict_proba(S_test)[:, 1]
    y_pred_lst.append(pred)
    y_pred_binary_lst.append(pred_to_binary(pred, threshold = threshold, per_of_zero=per_of_zero))

    
# Make 'output.csv'
final, final_df = export_csv(patient_num, error_patient, y_pred_binary_lst, y_pred_lst, path = path, index=final_idx)
print(making_result(S_test, y_pred_lst, y_pred_binary_lst, y_pred_lst2, y_pred_binary_lst2, include_model, include_model2, include_model3, final))
