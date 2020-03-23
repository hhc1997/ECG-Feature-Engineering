import pandas as pd
import numpy as np
import scipy.io as scio
import sys
import os
import wfdb
from scipy import signal
import matplotlib.pyplot as plt
#####采样后的新mat文件已经是除过ADC_gain了
def resampleECG(type_of_data,resampleFS,type2_file_name):

    def readname(filePath):
        name = os.listdir(filePath)
        return name
    file_colletion = readname(sys.path[0])
    if type_of_data > 0:
        dat_collection = []
        for i in range(0,len(file_colletion)):
            if file_colletion[i].find('.mat')>=0:
                dat_collection.append(file_colletion[i].strip('.mat'))
        for j in range(0,len(dat_collection)):
            record = wfdb.rdrecord(dat_collection[j])
            sampleNum = record.__dict__['sig_len']
            resample_num = int(sampleNum*(resampleFS/record.__dict__['fs']))
            data = pd.DataFrame(record.__dict__['p_signal'],columns=record.__dict__['sig_name'])
            resample_data_300HZ = pd.DataFrame()
            for k in record.__dict__['sig_name']:
                resample_data_300HZ[k] = signal.resample(data[k],resample_num,axis=0, window=None)
            scio.savemat(dat_collection[j]+'.mat',{'val':resample_data_300HZ.values.T})

    if type_of_data == 0:
        patient_collection = []
        for i in range(0,len(file_colletion)):
            if file_colletion[i].find(type2_file_name)>=0:
                patient_collection.append(file_colletion[i])
        for k in range(0,len(patient_collection)):
            concrete_file_colletion = readname(str(sys.path[0])+str('/')+str(patient_collection[k]))
            dat_collection = []
            for j in range(0,len(concrete_file_colletion)):
                if concrete_file_colletion[j].find('.mat')>=0:
                    dat_collection.append(concrete_file_colletion[j].strip('.dat'))
                    for l in range(0,len(dat_collection)):
                        record = wfdb.rdrecord(str(sys.path[0])+"/"+str(patient_collection[k])+"/"+dat_collection[l])
                        sampleNum = record.__dict__['sig_len']
                        resample_num = int(sampleNum*(resampleFS/record.__dict__['fs']))
                        data = pd.DataFrame(record.__dict__['p_signal'])
                        data.columns = record.__dict__['sig_name']
                        data = data['ii']
                        resample_data_300HZ = pd.DataFrame()
                        resample_data_300HZ['ECG1'] = signal.resample(data,resample_num,axis=0, window=None)
                        resample_data_300HZ.to_hdf('resample_300HZ_'+str(patient_collection[k])+str('_')+str(dat_collection[l])+'_.h5', key='resample_data_300HZ', mode='w')
resampleECG(True,300,None)