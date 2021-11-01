import numpy as np
import matplotlib.pyplot as plt
from Project1_helpers import *
from My_helpers import *
from project1_ridge import cross_validation_ridge
from project1_ridge import build_poly

print("loading the train set...\n")
DATA_TRAIN_PATH = 'train.csv'        # TODO: <<------ supply path here for train data
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
print("done..\n")
print("loading the test set...\n")
DATA_TEST_PATH = 'test.csv'         # TODO: <<------ supply path here for test data 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
print("done..\n")
OUTPUT_PATH = 'final_output.csv'    # TODO: <<------ supply path here for output csv


data_dict = matrix_to_dict (tX)
test_data_dict = matrix_to_dict (tX_test)

########## Creating jet wise split for train data ###################
print("Creating jet wise split for train data...\n")
j0_idx= np.where (tX[:,22]==0)[0]
j1_idx= np.where (tX[:,22]==1)[0]
j2_idx= np.where (tX[:,22]==2)[0]
j3_idx= np.where (tX[:,22]==3)[0]
######## Creating jet number wise splits ########### 
X_j0, y_j0 = jet_wise_split(j0_idx,tX,y)
X_j1, y_j1 = jet_wise_split(j1_idx,tX,y)
X_j2, y_j2 = jet_wise_split(j2_idx,tX,y)
X_j3, y_j3 = jet_wise_split(j3_idx,tX,y)

#############  Creating jet wise dict ######
data_dict_j0 = matrix_to_dict(X_j0)
data_dict_j1 = matrix_to_dict(X_j1)
data_dict_j2 = matrix_to_dict(X_j2)
data_dict_j3 = matrix_to_dict(X_j3)


########## Creating jet wise split for test data #########################################################################################
print("Creating jet wise split for test data...\n")
test_j0_idx= np.where (tX_test[:,22]==0)[0]
test_j1_idx= np.where (tX_test[:,22]==1)[0]
test_j2_idx= np.where (tX_test[:,22]==2)[0]
test_j3_idx= np.where (tX_test[:,22]==3)[0]


######## Creating jet number wise splits ########### 
X_test_j0, ids_test_j0 = jet_wise_split(test_j0_idx,tX_test,ids_test)
X_test_j1, ids_test_j1 = jet_wise_split(test_j1_idx,tX_test,ids_test)
X_test_j2, ids_test_j2 = jet_wise_split(test_j2_idx,tX_test,ids_test)
X_test_j3, ids_test_j3 = jet_wise_split(test_j3_idx,tX_test,ids_test)

#############  Creating jet wise dict ######
test_data_dict_j0 = matrix_to_dict(X_test_j0)
test_data_dict_j1 = matrix_to_dict(X_test_j1)
test_data_dict_j2 = matrix_to_dict(X_test_j2)
test_data_dict_j3 = matrix_to_dict(X_test_j3)


############# Identifying 0-var columns for train data ###################
no_var_col_j0 = Zero_variance_detector(data_dict_j0)
no_var_col_j1 = Zero_variance_detector(data_dict_j1)
no_var_col_j2 = Zero_variance_detector(data_dict_j2)
no_var_col_j3 = Zero_variance_detector(data_dict_j3)

print("Train : Total 0 Variance columns:\nj_0:{}\nj_1:{}\nj_2:{}\nj_3:{}".format(len(no_var_col_j0)-1,len(no_var_col_j1)-1,len(no_var_col_j2)-1,len(no_var_col_j3)-1))


############# Identifying 0-var columns for test data #########################################################################################

no_var_col_j0_test = Zero_variance_detector(test_data_dict_j0)
no_var_col_j1_test = Zero_variance_detector(test_data_dict_j1)
no_var_col_j2_test = Zero_variance_detector(test_data_dict_j2)
no_var_col_j3_test = Zero_variance_detector(test_data_dict_j3)

print("Test: Total 0 Variance columns:\nj_0:{}\nj_1:{}\nj_2:{}\nj_3 {}".format(len(no_var_col_j0_test)-1,len(no_var_col_j1_test)-1,len(no_var_col_j2_test)-1,len(no_var_col_j3_test)-1))

# removing the zero variance columns
data_j0 = dict_to_matrix(data_dict_j0)
data_j1 = dict_to_matrix(data_dict_j1)
data_j2 = dict_to_matrix(data_dict_j2)
data_j3 = dict_to_matrix(data_dict_j3)
#print(data_j0.shape,data_j1.shape,data_j2.shape,data_j3.shape)

data_j0_new = np.delete(data_j0, no_var_col_j0, axis=1)
data_j1_new = np.delete(data_j1, no_var_col_j1, axis=1)
data_j2_new = np.delete(data_j2, no_var_col_j2, axis=1)
data_j3_new = np.delete(data_j3, no_var_col_j3, axis=1)

#print(data_j0_new.shape,data_j1_new.shape,data_j2_new.shape,data_j3_new.shape)



# removing the zero variance columns test data #########################################################################################
data_j0_test = dict_to_matrix(test_data_dict_j0)
data_j1_test = dict_to_matrix(test_data_dict_j1)
data_j2_test = dict_to_matrix(test_data_dict_j2)
data_j3_test = dict_to_matrix(test_data_dict_j3)
#print(data_j0_test.shape,data_j1_test.shape,data_j2_test.shape,data_j3_test.shape)

data_j0_new_test = np.delete(data_j0_test, no_var_col_j0, axis=1)
data_j1_new_test = np.delete(data_j1_test, no_var_col_j1, axis=1)
data_j2_new_test = np.delete(data_j2_test, no_var_col_j2, axis=1)
data_j3_new_test = np.delete(data_j3_test, no_var_col_j3, axis=1)

#print(data_j0_new_test.shape,data_j1_new_test.shape,data_j2_new_test.shape,data_j3_new_test.shape)

# removing the outlier and replacing them with mean or median
outlier_col = [0]
data_dict_j0_fixed = start_999_handle(matrix_to_dict(data_j0_new),outlier_col,'median')
data_dict_j1_fixed = start_999_handle(matrix_to_dict(data_j1_new),outlier_col,'median')
data_dict_j2_fixed = start_999_handle(matrix_to_dict(data_j2_new),outlier_col,'median')
data_dict_j3_fixed = start_999_handle(matrix_to_dict(data_j3_new),outlier_col,'median')

#print(len(data_dict_j0_fixed),len(data_dict_j1_fixed),len(data_dict_j2_fixed),len(data_dict_j3_fixed))


########### Removing -999 values from test data #########################################################################################
outlier_col = [0]
data_dict_j0_fixed_test = start_999_handle(matrix_to_dict(data_j0_new_test),outlier_col,'median')
data_dict_j1_fixed_test = start_999_handle(matrix_to_dict(data_j1_new_test),outlier_col,'median')
data_dict_j2_fixed_test = start_999_handle(matrix_to_dict(data_j2_new_test),outlier_col,'median')
data_dict_j3_fixed_test = start_999_handle(matrix_to_dict(data_j3_new_test),outlier_col,'median')

#print(len(data_dict_j0_fixed_test),len(data_dict_j1_fixed_test),len(data_dict_j2_fixed_test),len(data_dict_j3_fixed_test))

data_dict_j0_fixed_norm={}
data_dict_j1_fixed_norm={}
data_dict_j2_fixed_norm={}
data_dict_j3_fixed_norm={}
for i in range (0,len(data_dict_j0_fixed)):
    data_dict_j0_fixed_norm[i] = mean_var_normalize(data_dict_j0_fixed[i])
for i in range (0,len(data_dict_j1_fixed)):   
    data_dict_j1_fixed_norm[i] = mean_var_normalize(data_dict_j1_fixed[i])
for i in range (0,len(data_dict_j2_fixed)):    
    data_dict_j2_fixed_norm[i] = mean_var_normalize(data_dict_j2_fixed[i])
for i in range (0,len(data_dict_j3_fixed)):    
    data_dict_j3_fixed_norm[i] = mean_var_normalize(data_dict_j3_fixed[i])
    
#print(len(data_dict_j0_fixed_norm),len(data_dict_j1_fixed_norm),len(data_dict_j2_fixed_norm),len(data_dict_j3_fixed_norm))


################# Normalizing the test data #########################################################################################
data_dict_j0_fixed_norm_test={}
data_dict_j1_fixed_norm_test={}
data_dict_j2_fixed_norm_test={}
data_dict_j3_fixed_norm_test={}
for i in range (0,len(data_dict_j0_fixed_test)):
    data_dict_j0_fixed_norm_test[i] = mean_var_normalize(data_dict_j0_fixed_test[i])
for i in range (0,len(data_dict_j1_fixed_test)):   
    data_dict_j1_fixed_norm_test[i] = mean_var_normalize(data_dict_j1_fixed_test[i])
for i in range (0,len(data_dict_j2_fixed_test)):    
    data_dict_j2_fixed_norm_test[i] = mean_var_normalize(data_dict_j2_fixed_test[i])
for i in range (0,len(data_dict_j3_fixed_test)):    
    data_dict_j3_fixed_norm_test[i] = mean_var_normalize(data_dict_j3_fixed_test[i])
    
#print(len(data_dict_j0_fixed_norm_test),len(data_dict_j1_fixed_norm_test),len(data_dict_j2_fixed_norm_test),len(data_dict_j3_fixed_norm_test))

### Matrices for training Models #####
data_j0_norm = dict_to_matrix(data_dict_j0_fixed_norm)
data_j1_norm = dict_to_matrix(data_dict_j1_fixed_norm)
data_j2_norm = dict_to_matrix(data_dict_j2_fixed_norm)
data_j3_norm = dict_to_matrix(data_dict_j3_fixed_norm)

# print("train J0-->",data_j0_norm.shape, y_j0.shape)
# print("train J1-->",data_j1_norm.shape, y_j1.shape)
# print("train J2-->",data_j2_norm.shape, y_j2.shape)
# print("train J3-->",data_j3_norm.shape, y_j3.shape)


############ creating the final test matriz for pridictions ####################
test_data_j0_norm = dict_to_matrix(data_dict_j0_fixed_norm_test)
test_data_j1_norm = dict_to_matrix(data_dict_j1_fixed_norm_test)
test_data_j2_norm = dict_to_matrix(data_dict_j2_fixed_norm_test)
test_data_j3_norm = dict_to_matrix(data_dict_j3_fixed_norm_test)

# print("test J0-->",test_data_j0_norm.shape, ids_test_j0.shape)
# print("test J1-->",test_data_j1_norm.shape, ids_test_j1.shape)
# print("test J2-->",test_data_j2_norm.shape, ids_test_j2.shape)
# print("test J3-->",test_data_j3_norm.shape, ids_test_j3.shape)

print("Data preprocessing for train test set complete\n\n")

train_ratio=0.8
x_tr_j0, y_tr_j0, x_ts_j0, y_ts_j0 =split_data(data_j0_norm, y_j0, train_ratio, seed=1)

x_tr_j1, y_tr_j1, x_ts_j1, y_ts_j1 =split_data(data_j1_norm, y_j1, train_ratio, seed=1)

x_tr_j2, y_tr_j2, x_ts_j2, y_ts_j2 =split_data(data_j2_norm, y_j2, train_ratio, seed=1)

x_tr_j3, y_tr_j3, x_ts_j3, y_ts_j3 =split_data(data_j3_norm, y_j3, train_ratio, seed=1)


# print("J0 -->", x_tr_j0.shape, y_tr_j0.shape, x_ts_j0.shape, y_ts_j0.shape)
# print("J1 -->", x_tr_j1.shape, y_tr_j1.shape, x_ts_j1.shape, y_ts_j1.shape)
# print("J2 -->", x_tr_j2.shape, y_tr_j2.shape, x_ts_j2.shape, y_ts_j2.shape)
# print("J3 -->", x_tr_j3.shape, y_tr_j3.shape, x_ts_j3.shape, y_ts_j3.shape)


seed = 32
k_fold = 4
lambdas = np.logspace(-10, 0, 30)

print("Training model for J0...\n")
ridge_rmse_tr_mean_j0, ridge_rmse_te_mean_j0, ridge_all_wts_mean_j0 = cross_validation_ridge(seed, k_fold,lambdas,y_tr_j0, x_tr_j0)
min_train_idx_j0= np.argmin(ridge_rmse_tr_mean_j0)
min_test_idx_j0= np.argmin(ridge_rmse_te_mean_j0)
ridg_wts_j0 = ridge_all_wts_mean_j0[min_test_idx_j0]
print("J0--> Min train loss= {:.2}(@ {:} iteration)  Min test loss= {:.2}(@ {:} iteration)\n\n".format(min(ridge_rmse_tr_mean_j0),min_train_idx_j0, min(ridge_rmse_te_mean_j0),min_test_idx_j0))

print("Training model for J1...\n")
ridge_rmse_tr_mean_j1, ridge_rmse_te_mean_j1, ridge_all_wts_mean_j1 = cross_validation_ridge(seed, k_fold,lambdas,y_tr_j1, x_tr_j1)
min_train_idx_j1= np.argmin(ridge_rmse_tr_mean_j1)
min_test_idx_j1= np.argmin(ridge_rmse_te_mean_j1)
ridg_wts_j1 = ridge_all_wts_mean_j1[min_test_idx_j1]
print("J1--> Min train loss= {:.2}(@ {:} iteration)  Min test loss= {:.2}(@ {:} iteration)\n\n".format(min(ridge_rmse_tr_mean_j1),min_train_idx_j1, min(ridge_rmse_te_mean_j1),min_test_idx_j1))

print("Training model for J2...\n")
ridge_rmse_tr_mean_j2, ridge_rmse_te_mean_j2, ridge_all_wts_mean_j2 = cross_validation_ridge(seed, k_fold,lambdas,y_tr_j2, x_tr_j2)
min_train_idx_j2= np.argmin(ridge_rmse_tr_mean_j2)
min_test_idx_j2= np.argmin(ridge_rmse_te_mean_j2)
ridg_wts_j2 = ridge_all_wts_mean_j2[min_test_idx_j2]
print("J2--> Min train loss= {:.2}(@ {:} iteration)  Min test loss= {:.2}(@ {:} iteration)\n\n".format(min(ridge_rmse_tr_mean_j2),min_train_idx_j2, min(ridge_rmse_te_mean_j2),min_test_idx_j2))

print("Training model for J3...\n")
ridge_rmse_tr_mean_j3, ridge_rmse_te_mean_j3, ridge_all_wts_mean_j3 = cross_validation_ridge(seed, k_fold,lambdas,y_tr_j3, x_tr_j3)
min_train_idx_j3= np.argmin(ridge_rmse_tr_mean_j3)
min_test_idx_j3= np.argmin(ridge_rmse_te_mean_j3)
ridg_wts_j3 = ridge_all_wts_mean_j2[min_test_idx_j3]
print("J3--> Min train loss= {:.2}(@ {:} iteration)  Min test loss= {:.2}(@ {:} iteration)\n\n".format(min(ridge_rmse_tr_mean_j3),min_train_idx_j3, min(ridge_rmse_te_mean_j3),min_test_idx_j3))


print("Optimal Lanbda values: \nj_0= {:.1e}\nj_1= {:.1e}\nj_2= {:.1e}\nj_3= {:.1e}".format(lambdas[min_test_idx_j0],lambdas[min_test_idx_j1],lambdas[min_test_idx_j2],lambdas[min_test_idx_j3]))

ridg_wts_mean_j0 = np.mean(np.array([ridg_wts_j0[0],ridg_wts_j0[1],ridg_wts_j0[2],ridg_wts_j0[3]]),axis=0)
ridg_wts_mean_j1 = np.mean(np.array([ridg_wts_j1[0],ridg_wts_j1[1],ridg_wts_j1[2],ridg_wts_j1[3]]),axis=0)
ridg_wts_mean_j2 = np.mean(np.array([ridg_wts_j2[0],ridg_wts_j2[1],ridg_wts_j2[2],ridg_wts_j2[3]]),axis=0)
ridg_wts_mean_j3 = np.mean(np.array([ridg_wts_j3[0],ridg_wts_j3[1],ridg_wts_j3[2],ridg_wts_j3[3]]),axis=0)

degree = 6
test_data_j0_norm_poly = build_poly (test_data_j0_norm, degree)
test_data_j1_norm_poly = build_poly (test_data_j1_norm, degree)
test_data_j2_norm_poly = build_poly (test_data_j2_norm, degree)
test_data_j3_norm_poly = build_poly (test_data_j3_norm, degree)

j0_pred = predict_labels(ridg_wts_mean_j0, test_data_j0_norm_poly)
j1_pred = predict_labels(ridg_wts_mean_j1, test_data_j1_norm_poly)
j2_pred = predict_labels(ridg_wts_mean_j2, test_data_j2_norm_poly)
j3_pred = predict_labels(ridg_wts_mean_j3, test_data_j3_norm_poly)

j0_pred_new = np.reshape(j0_pred,(j0_pred.shape[0],1))
j1_pred_new = np.reshape(j1_pred,(j1_pred.shape[0],1))
j2_pred_new = np.reshape(j2_pred,(j2_pred.shape[0],1))
j3_pred_new = np.reshape(j3_pred,(j3_pred.shape[0],1))

ids_test_j0_new = np.reshape(ids_test_j0,(ids_test_j0.shape[0],1))
ids_test_j1_new = np.reshape(ids_test_j1,(ids_test_j1.shape[0],1))
ids_test_j2_new = np.reshape(ids_test_j2,(ids_test_j2.shape[0],1))
ids_test_j3_new = np.reshape(ids_test_j3,(ids_test_j3.shape[0],1))

all_pred = np.vstack((j0_pred_new,j1_pred_new,j2_pred_new,j3_pred_new))
all_ids =  np.vstack((ids_test_j0_new,ids_test_j1_new,ids_test_j2_new,ids_test_j3_new))

print("generating the csv")
# TODO: fill in desired name of output file for submission
create_csv_submission(all_ids, all_pred, OUTPUT_PATH)
print("procedure complete")