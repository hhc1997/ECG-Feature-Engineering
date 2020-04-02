import sys
sys.path.append(r'c:\users\administrator\appdata\local\programs\python\python37\lib\site-packages')
import pandas as pd
from scipy import signal
import time
import os
import numpy as np
import biosppy
import scipy.io as scio
from scipy.signal import periodogram
import wfdb
from pyentrp import entropy as ent
import pywt
from QRS_features import sampen2
from QRS_features import SampleEn
from QRS_features import CDF
from QRS_features import CoeffOfVariation
from QRS_features import MAD
from QRS_features import QRSBasicStat
from QRS_features import QRSBasicStatDeltaRR
from QRS_features import QRSBasicStatPointMedian
from QRS_features import qrs_autocorr
from QRS_features import QRSYuxi
from QRS_features import Variability
from QRS_features import bin_stat
from QRS_features import get_columns
from utils import readname
from utils import WTfilt_1d

file_colletion= readname('resample_data'+str('/'))
dat_collection = []
###需要特征名请自行查找
QRS_feature = pd.DataFrame(columns=get_columns())
for i in range(0, len(file_colletion)):
    if file_colletion[i].find('.mat') >= 0:
        dat_collection.append(file_colletion[i])
for j in range(len(dat_collection)):
    record = wfdb.rdrecord('resample_data'+str('/')+dat_collection[j].strip('.mat'))
    data = scio.loadmat('resample_data'+str('/')+dat_collection[j])
    print('Data Read:'+time.strftime("%H:%M:%S"))
    ###以2导联为基准
    ecg_lead_2 = WTfilt_1d(data['val'][1])
    ###R峰检测
    rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal = ecg_lead_2,sampling_rate = 300)
    ###list代表 同一个病人的数据 切出的不同心拍
    QRS_list = []
    for each_R in range(1,len(rpeaks_indices_1[0])-1):
        R_P = rpeaks_indices_1[0][each_R]
        fs = 300
        ts = ecg_lead_2
        ini_point = R_P-90
        T_start = ini_point+round(0.4 * fs)
        T_end = ini_point+round(0.6 * fs)
        P_start = ini_point+round(0.1 * fs)
        P_end = ini_point+round(0.2 * fs)
        
        T_wave = ts[T_start:T_end]
        P_wave = ts[P_start:P_end]

        T_peak = max(T_wave)
        T_peak_idx = np.argmax(T_wave)+T_start

        P_peak = max(P_wave)
        P_peak_idx = np.argmax(P_wave)+P_start

        R_peak = ts[R_P]

        Q_peak = min(ts[P_end:R_P])
        Q_peak_idx = np.argmin(ts[P_end:R_P]) + P_end 

        S_peak = min(ts[R_P:T_start])
        S_peak_idx = np.argmin(ts[R_P:T_start]) + R_P
        QRS_ = ts[Q_peak_idx:S_peak_idx]
        if len(QRS_) > 1:
            QRS_list.append(len(QRS_))
    if  ',' not in record.__dict__['comments'][2][4:]:
        QRS_f_data = []
        QRS_f_data.extend([i for i in QRSBasicStat(ts)])
        QRS_f_data.extend([i for i in QRSBasicStatPointMedian(ts)])
        QRS_f_data.extend([i for i in QRSBasicStatDeltaRR(ts)])
        QRS_f_data.extend([i for i in QRSYuxi(ts)])
        QRS_f_data.extend([i for i in Variability(ts)])
        QRS_f_data.extend([i for i in bin_stat(ts)])
        QRS_f_data.extend([i for i in qrs_autocorr(ts)])
        temp_QRS_feature = pd.DataFrame(columns=get_columns(),data = [QRS_f_data])
        QRS_feature = QRS_feature.append(temp_QRS_feature)
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
            QRS_f_data = []
            QRS_f_data.extend([i for i in QRSBasicStat(ts)])
            QRS_f_data.extend([i for i in QRSBasicStatPointMedian(ts)])
            QRS_f_data.extend([i for i in QRSBasicStatDeltaRR(ts)])
            QRS_f_data.extend([i for i in QRSYuxi(ts)])
            QRS_f_data.extend([i for i in Variability(ts)])
            QRS_f_data.extend([i for i in bin_stat(ts)])
            QRS_f_data.extend([i for i in qrs_autocorr(ts)])
            temp_QRS_feature = pd.DataFrame(columns=get_columns(),data = [QRS_f_data])
            QRS_feature = QRS_feature.append(temp_QRS_feature)
    print('finish'+str(j))
QRS_feature.to_csv('FeatureData\QRS_FEATURE.csv', sep=',', header=True, index=False)


    