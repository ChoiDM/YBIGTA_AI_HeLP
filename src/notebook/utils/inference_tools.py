import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def pred_to_binary(pred_array, threshold = 0.5, per_of_zero=4):
    
    if threshold == "auto" :
        pred_binary = sorted(list(pred_array))
        threshold = pred_binary[int(len(pred_binary)*per_of_zero/10)]
        
        pred_binary = np.copy(pred_array)
        pred_binary[pred_binary > threshold] = 1
        pred_binary[pred_binary <= threshold] = 0
        
    else :
        pred_binary = np.copy(pred_array)
        pred_binary[pred_binary > threshold] = 1
        pred_binary[pred_binary <= threshold] = 0

    return pred_binary


def export_csv(patient_num, error_patient, y_pred_binary, y_pred, path="/data", index=None, class_of_error_patient=0):
    
    if index != None :
        y_pred_binary = y_pred_binary[index-1]
        y_pred = y_pred[index-1]

    values = [[num, binary, prob] for num, binary, prob in zip(patient_num, y_pred_binary, y_pred)]
    
    for patient in error_patient:
        if class_of_error_patient == 0:
            values.append([patient, class_of_error_patient, 0.0])

        elif class_of_error_patient == 1:
            values.append([patient, class_of_error_patient, 1.0])

    final_df = pd.DataFrame(values)
    final_df.to_csv(path+'/output/output.csv', sep = ',', header = False, index = False)
    
    return y_pred_binary, final_df


def making_result(S, y_pred_lst, y_pred_binary_lst, y_pred_lst2, y_pred_binary_lst2, models, stacking_models, stacking_models2, Y):
    
    if len(stacking_models) == 1 :
        values = [list(s)+[p0, pb0, pp0, pp1, y] 
                  for s, p0, pb0, pp0, pp1, y
                  in zip(S, 
                         y_pred_lst[0],
                         y_pred_binary_lst[0],
                         Y)]
        
    elif len(stacking_models) == 2 and len(stacking_models2) == 0 :
        values = [list(s)+[p0, p1, pb0, pb1, y] 
                  for s, p0, p1, pb0, pb1, y
                  in zip(S, 
                         y_pred_lst[0], y_pred_lst[1],
                         y_pred_binary_lst[0], y_pred_binary_lst[1],
                         Y)]
        
    elif len(stacking_models) == 2 and len(stacking_models2) == 1 :
        values = [list(s)+[p0, p1, pb0, pb1, pp0, ppb0,y] 
                  for s, p0, p1, pb0, pb1, pp0, ppb0, y
                  in zip(S, 
                         y_pred_lst[0], y_pred_lst[1],
                         y_pred_binary_lst[0], y_pred_binary_lst[1],
                         y_pred_lst2[0],
                         y_pred_binary_lst2[0],
                         Y)]
        
    elif len(stacking_models) == 2 and len(stacking_models2) == 2 :
        values = [list(s)+[p0, p1, pb0, pb1, pp0, pp1, ppb0, ppb1 ,y] 
                  for s, p0, p1, pb0, pb1, pp0, pp1, ppb0, ppb1,y
                  in zip(S, 
                         y_pred_lst[0], y_pred_lst[1],
                         y_pred_binary_lst[0], y_pred_binary_lst[1],
                         y_pred_lst2[0], y_pred_lst2[1],
                         y_pred_binary_lst2[0], y_pred_binary_lst2[1],
                         Y)]
        
    elif len(stacking_models) == 3 and len(stacking_models2) == 0 :
        values = [list(s)+[p0, p1, p2, pb0, pb1, pb2 ,y] 
                  for s, p0, p1, p2, pb0, pb1, pb2, y
                  in zip(S, 
                         y_pred_lst[0], y_pred_lst[1], y_pred_lst[2],
                         y_pred_binary_lst[0], y_pred_binary_lst[1], y_pred_binary_lst[2],
                         Y)]
        
    elif len(stacking_models) == 3 and len(stacking_models2) == 1 :
        values = [list(s)+[p0, p1, p2, pb0, pb1, pb2, pp0, ppb0 ,y] 
                  for s, p0, p1, p2, pb0, pb1, pb2, pp0, ppb0, y
                  in zip(S, 
                         y_pred_lst[0], y_pred_lst[1], y_pred_lst[2],
                         y_pred_binary_lst[0], y_pred_binary_lst[1], y_pred_binary_lst[2],
                         y_pred_lst2[0],
                         y_pred_binary_lst2[0],
                         Y)]
        
    elif len(stacking_models) == 3 and len(stacking_models2) == 2 :
        values = [list(s)+[p0, p1, p2, pb0, pb1, pb2, pp0, pp1, ppb0, ppb1 ,y] 
                  for s, p0, p1, p2, pb0, pb1, pb2, pp0, pp1, ppb0, ppb1,y
                  in zip(S, 
                         y_pred_lst[0], y_pred_lst[1], y_pred_lst[2],
                         y_pred_binary_lst[0], y_pred_binary_lst[1], y_pred_binary_lst[2],
                         y_pred_lst2[0], y_pred_lst2[1],
                         y_pred_binary_lst2[0], y_pred_binary_lst2[1],
                         Y)]
        
    elif len(stacking_models) == 4 and len(stacking_models2) == 0 :
        values = [list(s)+[p0, p1, p2, p3, pb0, pb1, pb2, pb3, y] 
                  for s, p0, p1, p2, p3, pb0, pb1, pb2, pb3, y
                  in zip(S, 
                         y_pred_lst[0], y_pred_lst[1], y_pred_lst[2], y_pred_lst[3],
                         y_pred_binary_lst[0], y_pred_binary_lst[1], y_pred_binary_lst[2], y_pred_binary_lst[3], 
                         Y)]
        
    elif len(stacking_models) == 4 and len(stacking_models2) == 1 :
        values = [list(s)+[p0, p1, p2, p3, pb0, pb1, pb2, pb3, pp0, ppb0, y] 
                  for s, p0, p1, p2, p3, pb0, pb1, pb2, pb3, pp0, ppb0, y
                  in zip(S, 
                         y_pred_lst[0], y_pred_lst[1], y_pred_lst[2], y_pred_lst[3],
                         y_pred_binary_lst[0], y_pred_binary_lst[1], y_pred_binary_lst[2], y_pred_binary_lst[3], 
                         y_pred_lst2[0],
                         y_pred_binary_lst2[0],
                         Y)]
        
    elif len(stacking_models) == 4 and len(stacking_models2) == 2 :
        values = [list(s)+[p0, p1, p2, p3, pb0, pb1, pb2, pb3, pp0, pp1, ppb0, ppb1 ,y] 
                  for s, p0, p1, p2, p3, pb0, pb1, pb2, pb3, pp0, pp1, ppb0, ppb1,y
                  in zip(S, 
                         y_pred_lst[0], y_pred_lst[1], y_pred_lst[2], y_pred_lst[3],
                         y_pred_binary_lst[0], y_pred_binary_lst[1], y_pred_binary_lst[2], y_pred_binary_lst[3], 
                         y_pred_lst2[0], y_pred_lst2[1],
                         y_pred_binary_lst2[0], y_pred_binary_lst2[1],
                         Y)]

    final_df = pd.DataFrame(values, columns = ["m_"+str(idx) for idx in models] 
                                               + ["stack_"+str(idx) for idx in stacking_models] 
                                               + ["stack_b_"+str(idx) for idx in stacking_models] 
                                               + ["stack2_"+str(idx) for idx in stacking_models2]
                                               + ["stack2_b_"+str(idx) for idx in stacking_models2]
                                               + ["Y"])
    return final_df
