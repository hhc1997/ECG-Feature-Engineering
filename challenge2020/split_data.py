import sys
import pandas as pd
import numpy as np
import scipy.io as scio
import os
from random import shuffle
def readname(filePath):
    name = os.listdir(filePath)
    return name
def get_train_val_test():

    file_colletion = readname('/data/hanhaochen/fixed_length_300HZ'+str('/'))
    dat_collection = []
    for i in range(0,len(file_colletion)):
        if file_colletion[i].find('.mat')>=0:
            dat_collection.append(file_colletion[i].strip('.mat'))
    shuffle(dat_collection)
    train_list = dat_collection[:5501]
    val_list = dat_collection[5501:6189]
    test_list = dat_collection[6189:]
    #train_list = dat_collection[:500]
    #val_list = dat_collection[600:700]
    #test_list = dat_collection[800:900]
    file = open('/home/hanhaochen/challenge2020_data/challenge2020/data_set/train.txt', 'w')
    file.write(str(train_list))
    file.close()

    file = open('/home/hanhaochen/challenge2020_data/challenge2020/data_set/val.txt', 'w')
    file.write(str(val_list))
    file.close()

    file = open('/home/hanhaochen/challenge2020_data/challenge2020/data_set/test.txt', 'w')
    file.write(str(test_list))
    file.close()

    return train_list,val_list,test_list

