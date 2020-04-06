# @Time : 2019/10/11 下午6:25 
# @Author : Xiaoyu Li
# @File : data.py 
# @Orgnization: Dr.Cubic Lab

import numpy as np
import scipy
import os
import pickle as dill
import scipy.io as scio
import wfdb
###########################
## Function to load data ##
##########################
def label_num(s):
    if s == 'AF':
        return 1
    elif s == 'I-AVB' :
        return 2
    elif s == 'LBBB' :
        return 3
    elif s == 'Normal' :
        return 4
    elif s == 'PAC' :
        return 5
    elif s == 'PVC' :
        return 6
    elif s == 'RBBB' :
        return 7
    elif s == 'STD' :
        return 8
    elif s == 'STE' :
        return 9
###########################
def labels_onehot(label):
    label_onehot = [0]*9
    label_onehot[int(label)-1] = 1
    return label_onehot
labels_onehot(1)
############################
def how_many_label(chars):
    num = 1
    for i in chars:
        if i == ',':
            num += 1
    return num
############################
def loaddata(train_list,test_list,validation_list):

    label_train = []
    sample_num_train = 0
    print(train_list)
    for i in range(len(train_list)):
        record = wfdb.rdrecord('/home/hanhaochen/challenge2020_data/resample_data' + str('/') + train_list[i])
        sample_num_train += (how_many_label(record.__dict__['comments'][2]))
    ###创建0矩阵 然后插入
    data_train = np.zeros((sample_num_train,43200,12))
    ###macth j to sample num
    j_match_sn = 0
    for j in range(len(train_list)):

        record = wfdb.rdrecord('/home/hanhaochen/challenge2020_data/resample_data' + str('/') + train_list[j])
        if ',' not in record.__dict__['comments'][2][4:]:
            temp_data_train = scio.loadmat('/data/hanhaochen/fixed_length_300HZ' + str('/') + train_list[j])['val'].T
            data_train[j+j_match_sn]= temp_data_train
            label_train.append(labels_onehot(label_num(record.__dict__['comments'][2][4:])))
        else:
            s = record.__dict__['comments'][2][4:]
            s += '0'
            while len(s) != 0:
                ans = ''
                for k in range(len(s)):
                    if s[k] != ',' and s[k] != '0':
                        ans += s[k]
                    elif s[k] == ',':
                        s = s[k + 1:]
                        break
                    elif s[k] == '0':
                        s = ''

                temp_data_train = scio.loadmat('/data/hanhaochen/fixed_length_300HZ' + str('/') + train_list[j])['val'].T
                data_train[j+j_match_sn]= temp_data_train
                label_train.append(labels_onehot(label_num(ans)))
                j_match_sn += 1
            j_match_sn -= 1
        print(j, 'of',len(train_list))
    print(data_train)
    print(data_train.shape)
    print(len(label_train))
    X_ecg = data_train
    y = label_train

    label_validation = []
    sample_num_val = 0
    for i in range(len(validation_list)):
        record = wfdb.rdrecord('/home/hanhaochen/challenge2020_data/resample_data' + str('/') + validation_list[i])
        sample_num_val += (how_many_label(record.__dict__['comments'][2]))
    ###创建0矩阵 然后插入
    data_validation = np.zeros((sample_num_val, 43200, 12))
    j_match_sn = 0
    for j in range(len(validation_list)):
        record = wfdb.rdrecord('/home/hanhaochen/challenge2020_data/resample_data' + str('/') + validation_list[j])
        if ',' not in record.__dict__['comments'][2][4:]:
            temp_data_validation = scio.loadmat('/data/hanhaochen/fixed_length_300HZ' + str('/') + validation_list[j])['val'].T
            data_validation[j+j_match_sn]= temp_data_validation
            label_validation.append(labels_onehot(label_num(record.__dict__['comments'][2][4:])))
        else:
            s = record.__dict__['comments'][2][4:]
            s += '0'
            while len(s) != 0:
                ans = ''
                for k in range(len(s)):
                    if s[k] != ',' and s[k] != '0':
                        ans += s[k]
                    elif s[k] == ',':
                        s = s[k + 1:]
                        break
                    elif s[k] == '0':
                        s = ''
                temp_data_validation = scio.loadmat('/data/hanhaochen/fixed_length_300HZ' + str('/') + validation_list[j])['val'].T
                data_validation[j + j_match_sn] = temp_data_validation
                label_validation.append(labels_onehot(label_num(ans)))
                j_match_sn += 1
            j_match_sn -= 1
        print(j,'of',len(validation_list))
    print(data_validation.shape)
    print(len(label_validation))
    val_set_ecg = data_validation
    val_target = label_validation


    label_test = []
    sample_num_test = 0
    for i in range(len(validation_list)):
        record = wfdb.rdrecord('/home/hanhaochen/challenge2020_data/resample_data' + str('/') + test_list[i])
        sample_num_test += (how_many_label(record.__dict__['comments'][2]))
    ###创建0矩阵 然后插入
    data_test = np.zeros((sample_num_test, 43200, 12))
    j_match_sn = 0
    for j in range(len(test_list)):
        record = wfdb.rdrecord('/home/hanhaochen/challenge2020_data/resample_data' + str('/') + test_list[j])
        if ',' not in record.__dict__['comments'][2][4:]:
            temp_data_test = scio.loadmat('/data/hanhaochen/fixed_length_300HZ' + str('/') + test_list[j])['val'].T
            data_test[j+j_match_sn] = temp_data_test
            label_test.append(labels_onehot(label_num(record.__dict__['comments'][2][4:])))
        else:
            s = record.__dict__['comments'][2][4:]
            s += '0'
            while len(s) != 0:
                ans = ''
                for k in range(len(s)):
                    if s[k] != ',' and s[k] != '0':
                        ans += s[k]
                    elif s[k] == ',':
                        s = s[k + 1:]
                        break
                    elif s[k] == '0':
                        s = ''
                temp_data_test = scio.loadmat('/data/hanhaochen/fixed_length_300HZ' + str('/') + test_list[j])['val'].T
                data_test[j + j_match_sn] = temp_data_test
                label_test.append(labels_onehot(label_num(ans)))
                j_match_sn += 1
            j_match_sn -= 1
        print(j,'of' ,len(test_list))
    print(data_test.shape)
    print(len(label_test))
    final_testset_ecg = data_test
    final_testtarget = label_test



    return (X_ecg, y), (val_set_ecg, val_target), (final_testset_ecg, final_testtarget)

