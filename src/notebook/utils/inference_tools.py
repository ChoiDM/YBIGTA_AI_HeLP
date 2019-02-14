import numpy as np
import pandas as pd


def pred_to_binary(pred_array, threshold = 0.5):

    pred_binary = np.copy(pred_array)
    pred_binary[pred_binary > threshold] = 1
    pred_binary[pred_binary <= threshold] = 0

    return pred_binary


def export_csv(patient_num, y_pred_binary, y_pred, path="/data/output/"):

    values = [[num, binary, prob] for num, binary, prob in
                zip(patient_num, y_pred_binary, y_pred)]

    final_df = pd.DataFrame(values)
    final_df.to_csv(path+'output.csv', sep = ',', header = False, index = False)
    
    return final_df

def making_df(S_train, meta, y_true):

    values = [list(s)+[m]+[y] for s,m,y in zip(S_train, meta, y_true)]

    final_df = pd.DataFrame(values, columns =["model1", "model3", "model4", "model5", "model10", "meta", "y_true"])
    return final_df