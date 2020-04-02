import sys
sys.path.append(r'c:\users\administrator\appdata\local\programs\python\python37\lib\site-packages')
import os
import wfdb
import pywt
import pandas as pd
from scipy import signal
import time
import os
import numpy as np

def readname(filePath):
    name = os.listdir(filePath)
    return name

def gender_num(s):
    if s == 'Male':
        return 1
    elif s == 'Female':
        return 0

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


def sample_length():
    sig_len_all = []
    def readname(filePath):
        name = os.listdir(filePath)
        return name
    file_colletion = readname('resample_data'+str('/'))
    dat_collection = []
    for i in range(0,len(file_colletion)):
        if file_colletion[i].find('.mat')>=0:
            dat_collection.append(file_colletion[i].strip('.mat'))
    for j in range(0,len(dat_collection)):
        record = wfdb.rdrecord('resample_data'+str('/')+dat_collection[j])
        sig_len_all.append(record.__dict__['sig_len']/record.__dict__['fs'])
    return sig_len_all

def flatten(x):
    result = []
    for i in x:
        for j in i:
            result.append(j)
    return result


def WTfilt_1d(sig):
    """
    # 使用小波变换对单导联ECG滤波
    # 参考：Martis R J, Acharya U R, Min L C. ECG beat classification using PCA, LDA, ICA and discrete
    wavelet transform[J].Biomedical Signal Processing and Control, 2013, 8(5): 437-448.
    :param sig: 1-D numpy Array，单导联ECG
    :return: 1-D numpy Array，滤波后信号
    """
    coeffs = pywt.wavedec(sig, 'db6', level=5)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt