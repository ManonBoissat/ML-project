################################################## Tilak's helper Functions #########################
import numpy as np
from Project1_helpers import predict_labels
def classification (wts, X, y):    
    y_ = predict_labels(wts, X)
    c=0
    for i in range (0, y.shape[0]):
        if y[i] == y_[i]:
            c+=1
    #false = y.shape[0]-c
    #print("classification Accuracy = {:.3}".format((c/y_.shape[0])*100))
    return c

def mean_var_normalize(data):
    z = (data - np.mean(data))/np.std(data)
    return z

def matrix_to_dict (data):
        data_dict ={}
        for i in range (0, data.shape[1]):
            data_dict[i] = data[:,i]
        return data_dict
    
def dict_to_matrix(data_dict):
    data_pts = len(data_dict[0])
    features = len(data_dict)
    matrix = np.zeros((data_pts,features))
    for i in range (0,features):
        matrix[:,i] =data_dict[i]
    return matrix


def jet_wise_split(idx_array,tX,y):
    X_j = np.zeros((idx_array.shape[0],30))
    y_j =np.zeros(idx_array.shape[0])

    for i in range (0,idx_array.shape[0]):
        X_j[i][:] = tX[idx_array[i]]
        y_j[i] = y[idx_array[i]]
    print(X_j.shape, y_j.shape)
    return X_j, y_j

def Zero_variance_detector(data_dict): 
    no_var =[]
    for i in range (0, len(data_dict)):
        if (np.var(data_dict[i]) ==0):
            no_var.append(i)
    return no_var


def otlier_999_handel (data_dict, col, stat):
    #print("for col :{}".format(col))
    if stat == 'mean':
        #print("Old mean = {} ; Old std = {} ".format(np.mean(data_dict[col]), np.std(data_dict[col])))
        temp = data_dict[col][~np.isin(data_dict[col], (-999))].copy()
        #print("good part mean = {} ; good part std = {} ".format(np.mean(temp), np.std(temp)))
        req_mean = np.mean(temp)
        col_dash = np.where(data_dict[col] == -999, req_mean, data_dict[col])
        #print("mean_dash = {} ; std_dash = {} ".format(np.mean(col_dash), np.std(col_dash)))
    
    if stat == 'median':
        #print("Old median = {} ; Old std = {} ".format(np.median(data_dict[col]), np.std(data_dict[col])))
        temp = data_dict[col][~np.isin(data_dict[col], (-999))].copy()
        #print("good part median = {} ; good part std = {} ".format(np.median(temp), np.std(temp)))
        req_median = np.median(temp)
        col_dash = np.where(data_dict[col] == -999, req_median, data_dict[col])
        #print("median_dash = {} ; std_dash = {} ".format(np.median(col_dash), np.std(col_dash)))
    
    return col_dash

def start_999_handle (data_dict, outlier_col, stat):
    data_dict_ = {}
    for i in range (0, len(data_dict)):
        if i in outlier_col:
            data_dict_[i] = otlier_999_handel(data_dict, i , stat)
        else:
            data_dict_[i] = data_dict[i]
    return data_dict_