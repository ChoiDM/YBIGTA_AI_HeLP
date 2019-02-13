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
    
    
    print('--------------------------------')
    print(final_df)
