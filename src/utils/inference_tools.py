import numpy as np
import pandas as pd
pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)


def pred_to_binary(pred_array, threshold = 0.5):
    
    if threshold == "auto" :
        pred_binary = sorted(list(pred_array))
        threshold = pred_binary[int(len(pred_binary)*6/10)]
        
        pred_binary = np.array(pred_binary)
        pred_binary[pred_binary > threshold] = 1
        pred_binary[pred_binary <= threshold] = 0
        
    else :
        pred_binary = np.copy(pred_array)
        pred_binary[pred_binary > threshold] = 1
        pred_binary[pred_binary <= threshold] = 0

    return pred_binary


def export_csv(patient_num, y_pred_binary, y_pred, path="/data", index=None):
    
    if index != None :
        y_pred_binary = y_pred_binary[index]
        y_pred = y_pred[index]

    values = [[num, binary, prob] for num, binary, prob in
                zip(patient_num, y_pred_binary, y_pred)]

    final_df = pd.DataFrame(values)
    final_df.to_csv(path+'/output/output.csv', sep = ',', header = False, index = False)
    
    return y_pred_binary, final_df


def making_result(S_train, y_pred_lst, y_pred_binary_lst, y_pred_lst2, y_pred_binary_lst2, Y, models=[1,3,4,5,7,10]):

    values = [list(s)+[p0, p1, p2, p3, pb0, pb1, pb2, pb3, pp0, pp1, pp2, pp3, ppb0, ppb1, ppb2, ppb3 ,y] 
              for s, p0, p1, p2, p3, pb0, pb1, pb2, pb3, pp0, pp1, pp2, pp3, ppb0, ppb1, ppb2, ppb3 ,y
              in zip(S_train, 
                     y_pred_lst[0], y_pred_lst[1], y_pred_lst[2], y_pred_lst[3],
                     y_pred_binary_lst[0], y_pred_binary_lst[1], y_pred_binary_lst[2], y_pred_binary_lst[3], 
                     y_pred_lst2[0], y_pred_lst2[1], y_pred_lst2[2], y_pred_lst2[3],
                     y_pred_binary_lst2[0], y_pred_binary_lst2[1], y_pred_binary_lst2[2], y_pred_binary_lst2[3], 
                     Y)]

    final_df = pd.DataFrame(values, columns = ["m"+str(idx) for idx in models] 
                                               + ["xgb", "lr", "NN", "weight", "xgb_b", "lr_b", "NN_b", "weight_b"]
                                               + ["xgb2", "lr2", "NN2", "weight2", "xgb_b2", "lr_b2", "NN_b2", "weight_b2","y"])
    return final_df
