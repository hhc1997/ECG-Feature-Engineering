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
from SHORT_features import LongThresCrossing
from utils import readname
from utils import WTfilt_1d

file_colletion= readname('resample_data'+str('/'))
dat_collection = []
###需要特征名请自行查找
SHORT_feature = pd.DataFrame(columns=['ShortStatWaveFeature_'+str(i) for i in range((2+5+16+2) * 6)])
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
    QRS_peak_list = []
    QRS_area_list = []
    PR_interval_list = []
    QRS_duration_list = []
    QT_interval_list = []
    QT_corrected_list = []
    RQ_amp_list = []
    RS_amp_list = []
    ST_amp_list = []
    PQ_amp_list = []
    QS_amp_list = []
    RP_amp_list = []
    RT_amp_list = []
    ST_interval_list = []
    RS_interval_list = []
    T_peak_list = []
    P_peak_list = []
    Q_peak_list = []
    R_peak_list = []
    S_peak_list = []
    RS_slope_list = []
    ST_slope_list = []
    NF_list = []
    Fwidth_list = []
    vent_rate_list = 60 / (np.diff(rpeaks_indices_1[0]) /300)
    for each_R in range(1,len(rpeaks_indices_1[0])-1):
        R_P = rpeaks_indices_1[0][each_R]
        fs = 300
        ts = ecg_lead_2
        ini_point = R_P-90
        T_start = ini_point+round(0.4 * fs)
        T_end = ini_point+round(0.6 * fs)
        P_start = ini_point+round(0.1 * fs)
        P_end = ini_point+round(0.2 * fs)
        #print(each_R)

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

        ### features, recent add (2)
        QRS_peak = R_peak  ##这里之前写的等于max(ts)  按理说max(ts)就是R峰 没必要再求一次max()
        QRS_area = np.sum(np.abs(ts[Q_peak_idx: R_P])) + np.sum(np.abs(ts[R_P: S_peak_idx]))
        ### features (5)
        PR_interval = R_P - P_peak_idx
        QRS_duration = S_peak_idx  - Q_peak_idx 
        QT_interval = T_peak_idx - Q_peak_idx
        QT_corrected = QT_interval / len(ts)
        ### number of f waves (2)
        TQ_interval = ts[Q_peak_idx:T_peak_idx]
        thres = np.mean(TQ_interval) + (T_peak - np.mean(TQ_interval))/50
        NF, Fwidth = LongThresCrossing(TQ_interval, thres)
        ### more features (16)
        RQ_amp = R_peak - Q_peak
        RS_amp = R_peak - S_peak
        ST_amp = T_peak - S_peak
        PQ_amp = P_peak - Q_peak
        QS_amp = Q_peak - S_peak
        RP_amp = R_peak - P_peak
        RT_amp = R_peak - T_peak
        ST_interval = T_peak_idx - S_peak_idx
        RS_interval = S_peak_idx - R_P
        if RS_interval == 0:
            RS_slope = 0
        else:
            RS_slope = RS_amp / RS_interval                
        if ST_interval == 0:
            ST_slope = 0
        else:
            ST_slope = ST_amp / ST_interval
        
        
        ### add to list
        QRS_peak_list.append(QRS_peak)
        QRS_area_list.append(QRS_area)
        PR_interval_list.append(PR_interval)
        QRS_duration_list.append(QRS_duration)
        QT_interval_list.append(QT_interval)
        QT_corrected_list.append(QT_corrected)
        NF_list.append(NF)
        Fwidth_list.append(Fwidth)
        RQ_amp_list.append(RQ_amp)
        RS_amp_list.append(RS_amp)
        ST_amp_list.append(ST_amp)
        PQ_amp_list.append(PQ_amp)
        QS_amp_list.append(QS_amp)
        RP_amp_list.append(RP_amp)
        RT_amp_list.append(RT_amp)
        ST_interval_list.append(ST_interval)
        RS_interval_list.append(RS_interval)
        T_peak_list.append(T_peak)
        P_peak_list.append(P_peak)
        Q_peak_list.append(Q_peak)
        R_peak_list.append(R_peak)
        S_peak_list.append(S_peak)
        RS_slope_list.append(RS_slope)
        ST_slope_list.append(ST_slope)
    if  ',' not in record.__dict__['comments'][2][4:]:
        temp_SHORT_feature = pd.DataFrame(data = [[np.mean(QRS_peak_list), 
                                np.mean(QRS_area_list), 
                                np.mean(PR_interval_list), 
                                np.mean(QRS_duration_list), 
                                np.mean(QT_interval_list), 
                                np.mean(QT_corrected_list), 
                                np.mean(RQ_amp_list),
                                np.mean(RS_amp_list),
                                np.mean(ST_amp_list),
                                np.mean(PQ_amp_list),
                                np.mean(QS_amp_list),
                                np.mean(RP_amp_list),
                                np.mean(RT_amp_list),
                                np.mean(ST_interval_list),
                                np.mean(RS_interval_list),
                                np.mean(T_peak_list),
                                np.mean(P_peak_list),
                                np.mean(Q_peak_list),
                                np.mean(R_peak_list),
                                np.mean(S_peak_list),
                                np.mean(RS_slope_list),
                                np.mean(ST_slope_list),
                                np.mean(NF_list),
                                np.mean(Fwidth_list),
                                np.mean(vent_rate_list), 
                                
                                np.max(QRS_peak_list), 
                                np.max(QRS_area_list), 
                                np.max(PR_interval_list), 
                                np.max(QRS_duration_list), 
                                np.max(QT_interval_list), 
                                np.max(QT_corrected_list), 
                                np.max(RQ_amp_list),
                                np.max(RS_amp_list),
                                np.max(ST_amp_list),
                                np.max(PQ_amp_list),
                                np.max(QS_amp_list),
                                np.max(RP_amp_list),
                                np.max(RT_amp_list),
                                np.max(ST_interval_list),
                                np.max(RS_interval_list),
                                np.max(T_peak_list),
                                np.max(P_peak_list),
                                np.max(Q_peak_list),
                                np.max(R_peak_list),
                                np.max(S_peak_list),
                                np.max(RS_slope_list),
                                np.max(ST_slope_list),
                                np.max(NF_list),
                                np.max(Fwidth_list), 
                                np.max(vent_rate_list),
                                
                                np.min(QRS_peak_list), 
                                np.min(QRS_area_list), 
                                np.min(PR_interval_list), 
                                np.min(QRS_duration_list), 
                                np.min(QT_interval_list), 
                                np.min(QT_corrected_list), 
                                np.min(RQ_amp_list),
                                np.min(RS_amp_list),
                                np.min(ST_amp_list),
                                np.min(PQ_amp_list),
                                np.min(QS_amp_list),
                                np.min(RP_amp_list),
                                np.min(RT_amp_list),
                                np.min(ST_interval_list),
                                np.min(RS_interval_list),
                                np.min(T_peak_list),
                                np.min(P_peak_list),
                                np.min(Q_peak_list),
                                np.min(R_peak_list),
                                np.min(S_peak_list),
                                np.min(RS_slope_list),
                                np.min(ST_slope_list),
                                np.min(NF_list),
                                np.min(Fwidth_list), 
                                np.min(vent_rate_list),
                                
                                np.std(QRS_peak_list), 
                                np.std(QRS_area_list), 
                                np.std(PR_interval_list), 
                                np.std(QRS_duration_list), 
                                np.std(QT_interval_list), 
                                np.std(QT_corrected_list), 
                                np.std(RQ_amp_list),
                                np.std(RS_amp_list),
                                np.std(ST_amp_list),
                                np.std(PQ_amp_list),
                                np.std(QS_amp_list),
                                np.std(RP_amp_list),
                                np.std(RT_amp_list),
                                np.std(ST_interval_list),
                                np.std(RS_interval_list),
                                np.std(T_peak_list),
                                np.std(P_peak_list),
                                np.std(Q_peak_list),
                                np.std(R_peak_list),
                                np.std(S_peak_list),
                                np.std(RS_slope_list),
                                np.std(ST_slope_list),
                                np.std(NF_list),
                                np.std(Fwidth_list), 
                                np.std(vent_rate_list),
                                
                                np.percentile(QRS_peak_list, 25), 
                                np.percentile(QRS_area_list, 25), 
                                np.percentile(PR_interval_list, 25), 
                                np.percentile(QRS_duration_list, 25), 
                                np.percentile(QT_interval_list, 25), 
                                np.percentile(QT_corrected_list, 25), 
                                np.percentile(RQ_amp_list, 25),
                                np.percentile(RS_amp_list, 25),
                                np.percentile(ST_amp_list, 25),
                                np.percentile(PQ_amp_list, 25),
                                np.percentile(QS_amp_list, 25),
                                np.percentile(RP_amp_list, 25),
                                np.percentile(RT_amp_list, 25),
                                np.percentile(ST_interval_list, 25),
                                np.percentile(RS_interval_list, 25),
                                np.percentile(T_peak_list, 25),
                                np.percentile(P_peak_list, 25),
                                np.percentile(Q_peak_list, 25),
                                np.percentile(R_peak_list, 25),
                                np.percentile(S_peak_list, 25),
                                np.percentile(RS_slope_list, 25),
                                np.percentile(ST_slope_list, 25),
                                np.percentile(NF_list, 25),
                                np.percentile(Fwidth_list, 25), 
                                np.percentile(vent_rate_list,25),
                                
                                np.percentile(QRS_peak_list, 75), 
                                np.percentile(QRS_area_list, 75), 
                                np.percentile(PR_interval_list, 75), 
                                np.percentile(QRS_duration_list, 75), 
                                np.percentile(QT_interval_list, 75), 
                                np.percentile(QT_corrected_list, 75), 
                                np.percentile(RQ_amp_list, 75),
                                np.percentile(RS_amp_list, 75),
                                np.percentile(ST_amp_list, 75),
                                np.percentile(PQ_amp_list, 75),
                                np.percentile(QS_amp_list, 75),
                                np.percentile(RP_amp_list, 75),
                                np.percentile(RT_amp_list, 75),
                                np.percentile(ST_interval_list, 75),
                                np.percentile(RS_interval_list, 75),
                                np.percentile(T_peak_list, 75),
                                np.percentile(P_peak_list, 75),
                                np.percentile(Q_peak_list, 75),
                                np.percentile(R_peak_list, 75),
                                np.percentile(S_peak_list, 75),
                                np.percentile(RS_slope_list, 75),
                                np.percentile(ST_slope_list, 75),
                                np.percentile(NF_list, 75),
                                np.percentile(Fwidth_list, 75),
                                np.percentile(vent_rate_list,75),
                                ]],columns=['ShortStatWaveFeature_'+str(i) for i in range((2+5+16+2) * 6)])
        SHORT_feature = SHORT_feature.append(temp_SHORT_feature)


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
            temp_SHORT_feature = pd.DataFrame(data = [[np.mean(QRS_peak_list), 
                                np.mean(QRS_area_list), 
                                np.mean(PR_interval_list), 
                                np.mean(QRS_duration_list), 
                                np.mean(QT_interval_list), 
                                np.mean(QT_corrected_list), 
                                np.mean(RQ_amp_list),
                                np.mean(RS_amp_list),
                                np.mean(ST_amp_list),
                                np.mean(PQ_amp_list),
                                np.mean(QS_amp_list),
                                np.mean(RP_amp_list),
                                np.mean(RT_amp_list),
                                np.mean(ST_interval_list),
                                np.mean(RS_interval_list),
                                np.mean(T_peak_list),
                                np.mean(P_peak_list),
                                np.mean(Q_peak_list),
                                np.mean(R_peak_list),
                                np.mean(S_peak_list),
                                np.mean(RS_slope_list),
                                np.mean(ST_slope_list),
                                np.mean(NF_list),
                                np.mean(Fwidth_list), 
                                np.mean(vent_rate_list), 
                                
                                np.max(QRS_peak_list), 
                                np.max(QRS_area_list), 
                                np.max(PR_interval_list), 
                                np.max(QRS_duration_list), 
                                np.max(QT_interval_list), 
                                np.max(QT_corrected_list), 
                                np.max(RQ_amp_list),
                                np.max(RS_amp_list),
                                np.max(ST_amp_list),
                                np.max(PQ_amp_list),
                                np.max(QS_amp_list),
                                np.max(RP_amp_list),
                                np.max(RT_amp_list),
                                np.max(ST_interval_list),
                                np.max(RS_interval_list),
                                np.max(T_peak_list),
                                np.max(P_peak_list),
                                np.max(Q_peak_list),
                                np.max(R_peak_list),
                                np.max(S_peak_list),
                                np.max(RS_slope_list),
                                np.max(ST_slope_list),
                                np.max(NF_list),
                                np.max(Fwidth_list), 
                                np.max(vent_rate_list), 
                                
                                np.min(QRS_peak_list), 
                                np.min(QRS_area_list), 
                                np.min(PR_interval_list), 
                                np.min(QRS_duration_list), 
                                np.min(QT_interval_list), 
                                np.min(QT_corrected_list), 
                                np.min(RQ_amp_list),
                                np.min(RS_amp_list),
                                np.min(ST_amp_list),
                                np.min(PQ_amp_list),
                                np.min(QS_amp_list),
                                np.min(RP_amp_list),
                                np.min(RT_amp_list),
                                np.min(ST_interval_list),
                                np.min(RS_interval_list),
                                np.min(T_peak_list),
                                np.min(P_peak_list),
                                np.min(Q_peak_list),
                                np.min(R_peak_list),
                                np.min(S_peak_list),
                                np.min(RS_slope_list),
                                np.min(ST_slope_list),
                                np.min(NF_list),
                                np.min(Fwidth_list), 
                                np.min(vent_rate_list), 
                                
                                np.std(QRS_peak_list), 
                                np.std(QRS_area_list), 
                                np.std(PR_interval_list), 
                                np.std(QRS_duration_list), 
                                np.std(QT_interval_list), 
                                np.std(QT_corrected_list), 
                                np.std(RQ_amp_list),
                                np.std(RS_amp_list),
                                np.std(ST_amp_list),
                                np.std(PQ_amp_list),
                                np.std(QS_amp_list),
                                np.std(RP_amp_list),
                                np.std(RT_amp_list),
                                np.std(ST_interval_list),
                                np.std(RS_interval_list),
                                np.std(T_peak_list),
                                np.std(P_peak_list),
                                np.std(Q_peak_list),
                                np.std(R_peak_list),
                                np.std(S_peak_list),
                                np.std(RS_slope_list),
                                np.std(ST_slope_list),
                                np.std(NF_list),
                                np.std(Fwidth_list), 
                                np.std(vent_rate_list), 
                                
                                np.percentile(QRS_peak_list, 25), 
                                np.percentile(QRS_area_list, 25), 
                                np.percentile(PR_interval_list, 25), 
                                np.percentile(QRS_duration_list, 25), 
                                np.percentile(QT_interval_list, 25), 
                                np.percentile(QT_corrected_list, 25), 
                                np.percentile(RQ_amp_list, 25),
                                np.percentile(RS_amp_list, 25),
                                np.percentile(ST_amp_list, 25),
                                np.percentile(PQ_amp_list, 25),
                                np.percentile(QS_amp_list, 25),
                                np.percentile(RP_amp_list, 25),
                                np.percentile(RT_amp_list, 25),
                                np.percentile(ST_interval_list, 25),
                                np.percentile(RS_interval_list, 25),
                                np.percentile(T_peak_list, 25),
                                np.percentile(P_peak_list, 25),
                                np.percentile(Q_peak_list, 25),
                                np.percentile(R_peak_list, 25),
                                np.percentile(S_peak_list, 25),
                                np.percentile(RS_slope_list, 25),
                                np.percentile(ST_slope_list, 25),
                                np.percentile(NF_list, 25),
                                np.percentile(Fwidth_list, 25), 
                                np.percentile(vent_rate_list,25), 
                                
                                np.percentile(QRS_peak_list, 75), 
                                np.percentile(QRS_area_list, 75), 
                                np.percentile(PR_interval_list, 75), 
                                np.percentile(QRS_duration_list, 75), 
                                np.percentile(QT_interval_list, 75), 
                                np.percentile(QT_corrected_list, 75), 
                                np.percentile(RQ_amp_list, 75),
                                np.percentile(RS_amp_list, 75),
                                np.percentile(ST_amp_list, 75),
                                np.percentile(PQ_amp_list, 75),
                                np.percentile(QS_amp_list, 75),
                                np.percentile(RP_amp_list, 75),
                                np.percentile(RT_amp_list, 75),
                                np.percentile(ST_interval_list, 75),
                                np.percentile(RS_interval_list, 75),
                                np.percentile(T_peak_list, 75),
                                np.percentile(P_peak_list, 75),
                                np.percentile(Q_peak_list, 75),
                                np.percentile(R_peak_list, 75),
                                np.percentile(S_peak_list, 75),
                                np.percentile(RS_slope_list, 75),
                                np.percentile(ST_slope_list, 75),
                                np.percentile(NF_list, 75),
                                np.percentile(Fwidth_list, 75),
                                np.percentile(vent_rate_list,75), 
                                ]],columns=['ShortStatWaveFeature_'+str(i) for i in range((2+5+16+2) * 6)])
            SHORT_feature = SHORT_feature.append(temp_SHORT_feature)
    print('finish'+str(j))
SHORT_feature.to_csv('FeatureData\SHORT_FEATURE.csv', sep=',', header=True, index=False)