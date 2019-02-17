import numpy as np
import pandas as pd


def pred_to_binary(pred_array, threshold = 0.5):

    pred_binary = np.copy(pred_array)
    pred_binary[pred_binary > threshold] = 1
    pred_binary[pred_binary <= threshold] = 0

    return pred_binary


def export_csv(patient_num, error_patient, class_of_error_patient, y_pred_binary, y_pred, path="/data/output/"):

    values = [[int(num), binary, prob] for num, binary, prob in
                zip(patient_num, y_pred_binary, y_pred)]

    for patient in error_patient:
        if class_of_error_patient == 0:
            values.append([int(patient), class_of_error_patient, 0.0])

        elif class_of_error_patient == 1:
            values.append([int(patient), class_of_error_patient, 1.0])

        else:
            print("parameter 'class_of_error_patient' should be 0 or 1.")

    final_df = pd.DataFrame(values, columns = ['Num', 'Binary', 'Prob'])
    final_df.sort_values(by = 'Num')

    # Save
    final_df.to_csv(path+'output.csv', sep = ',', header = False, index = False)
    
    
    print('--------------------------------')
    print(final_df)
