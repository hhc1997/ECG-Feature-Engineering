import pandas as pd
from scipy import signal
import sys
sys.path.append(r'c:\users\administrator\appdata\local\programs\python\python37\lib\site-packages')
import os
import numpy as np
from scipy import stats
from scipy.signal import periodogram
import math
import wfdb
from pyentrp import entropy as ent
import pywt



# ‘转弯’的个数
def zigzag(ts):
    '''
    number of zigzag
    '''

    num_zigzag = 1
    for i in range(len(ts)-2):
        num_1 = ts[i]
        num_2 = ts[i+1]
        num_3 = ts[i+2]
        if (num_2 - num_1) * (num_3 - num_2) < 0:
            num_zigzag += 1
    return num_zigzag

###自回归系数 t应该是t*fs t取1到12
def autocorr(ts, t):
    return np.corrcoef(np.array([ts[0:len(ts)-t], ts[t:len(ts)]]))[0,1]

def LongBasicStat(ts,feature_name,idx):
    '''
    ### 1
    TODO: 
    
    1. why too much features will decrease F1
    2. how about add them and feature filter before xgb
    
    '''


    Range = max(ts) - min(ts)
    Var = np.var(ts)
    Skew = stats.skew(ts)
    Kurtosis = stats.kurtosis(ts)
    Median = np.median(ts)
#    p_001 = np.percentile(ts, 0.01)
#    p_002 = np.percentile(ts, 0.02)
#    p_005 = np.percentile(ts, 0.05)
#    p_01 = np.percentile(ts, 0.1)
#    p_02 = np.percentile(ts, 0.2)
#    p_05 = np.percentile(ts, 0.5)
    p_1 = np.percentile(ts, 1)
#    p_2 = np.percentile(ts, 2)
    p_5 = np.percentile(ts, 5)
    p_10 = np.percentile(ts, 10)
    p_25 = np.percentile(ts, 25)
    p_75 = np.percentile(ts, 75)
    p_90 = np.percentile(ts, 90)
    p_95 = np.percentile(ts, 95)
#    p_98 = np.percentile(ts, 98)
    p_99 = np.percentile(ts, 99)
#    p_995 = np.percentile(ts, 99.5)
#    p_998 = np.percentile(ts, 99.8)
#    p_999 = np.percentile(ts, 99.9)
#    p_9995 = np.percentile(ts, 99.95)
#    p_9998 = np.percentile(ts, 99.98)
#    p_9999 = np.percentile(ts, 99.99)

    range_99_1 = p_99 - p_1
    range_95_5 = p_95 - p_5
    range_90_10 = p_90 - p_10
    range_75_25 = p_75 - p_25
    
#    return [Range, Var, Skew, Kurtosis, Median]

#    return [Range, Var, Skew, Kurtosis, Median, 
#            p_1, p_5, p_95, p_99]
    
    feature_name.extend(['LongBasicStat_Range'+'lead'+str(idx), 
                         'LongBasicStat_Var'+'lead'+str(idx), 
                        'LongBasicStat_Skew'+'lead'+str(idx), 
                        'LongBasicStat_Kurtosis'+'lead'+str(idx), 
                        'LongBasicStat_Median'+'lead'+str(idx), 
                        'LongBasicStat_p_1'+'lead'+str(idx), 
                        'LongBasicStat_p_5'+'lead'+str(idx), 
                        'LongBasicStat_p_95'+'lead'+str(idx), 
                        'LongBasicStat_p_99'+'lead'+str(idx), 
                        'LongBasicStat_p_10'+'lead'+str(idx), 
                        'LongBasicStat_p_25'+'lead'+str(idx), 
                        'LongBasicStat_p_75'+'lead'+str(idx), 
                        'LongBasicStat_p_90'+'lead'+str(idx), 
                        'LongBasicStat_range_99_1'+'lead'+str(idx), 
                        'LongBasicStat_range_95_5'+'lead'+str(idx), 
                        'LongBasicStat_range_90_10'+'lead'+str(idx), 
                        'LongBasicStat_range_75_25'+'lead'+str(idx)])
    return [Range, Var, Skew, Kurtosis, Median, 
            p_1, p_5, p_95, p_99, 
            p_10, p_25, p_75, p_90, 
            range_99_1, range_95_5, range_90_10, range_75_25]
    

def LongZeroCrossing(ts, thres,feature_name,idx):
    '''
    ### 2
    '''

    cnt = 0
    for i in range(len(ts)-1):
        if (ts[i] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
        if ts[i] == thres and (ts[i-1] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
    feature_name.extend(['LongZeroCrossing_cnt'+'lead'+str(idx)])
    return [cnt]
    
def LongFFTBandPower(ts,feature_name,idx):
    '''
    ### 3
    return list of power of each freq band
    
    TODO: different band cut method
    '''

    fs = 300
    nfft = len(ts)
    partition = [0, 1.5, 4, 8, 20, 100, fs/2]
    f, psd = periodogram(ts, fs)
    partition = [int(x * nfft / fs) for x in partition]
    p = [sum(psd[partition[x] : partition[x + 1]]) for x in range(len(partition)-1)]
    
    feature_name.extend(['LongFFTBandPower_'+'lead'+str(idx)+str(i) for i in range(len(p))])

    return p

def LongFFTPower(ts,feature_name,idx):
    '''
    ### 4
    return power
    
    no effect
    '''

    psd = periodogram(ts, fs=300.0, nfft=4500)
    power = np.sum(psd[1])
    feature_name.extend(['LongFFTPower_power'+'lead'+str(idx)])
    return [power]

def LongFFTBandPowerShannonEntropy(ts,feature_name,idx):
    '''
    ### 5
    return entropy of power of each freq band
    refer to scipy.signal.periodogram
    
    TODO: different band cut method
    '''

    fs = 300
    nfft = len(ts)
    partition = [0, 1.5, 4, 8, 20, 100, fs/2]
    f, psd = periodogram(ts, fs)
    partition = [int(x * nfft / fs) for x in partition]
    p = [sum(psd[partition[x] : partition[x + 1]]) for x in range(len(partition)-1)]
    prob = [x / sum(p) for x in p]
    entropy = sum([- x * math.log(x) for x in prob])
    feature_name.extend(['LongFFTBandPowerShannonEntropy_entropy'+'lead'+str(idx)])
    return [entropy]

def LongSNR(ts,feature_name,idx):
    '''
    ### 6
    TODO
    '''

    psd = periodogram(ts, fs=300.0)

    signal_power = 0
    noise_power = 0
    for i in range(len(psd[0])):
        if psd[0][i] < 5:
            signal_power += psd[1][i]
        else:
            noise_power += psd[1][i]
    
    feature_name.extend(['LongSNR_snr'+'lead'+str(idx)])
      
    return [signal_power / noise_power]

def long_autocorr(ts,feature_name,idx):
    '''
    ### 7
    '''
    feat = []
    num_lag = 12
    
    feature_name.extend(['long_autocorr_'+str(idx)+'lead'+str(i) for i in range(num_lag)])
    
    for i in range(num_lag):
        feat.append(autocorr(ts, i))

    return feat

def long_zigzag(ts,feature_name,idx):
    '''
    ### 8
    '''
    feature_name.extend(['long_zigzag'+'lead'+str(idx)])
    num_zigzag = zigzag(ts)
    return [num_zigzag]

def LongThresCrossing(ts,feature_name,idx):
    '''
    ### 9
    '''
    thres = np.mean(ts)

    cnt = 0
    pair_flag = 1
    pre_loc = 0
    width = []
    for i in range(len(ts)-1):
        if (ts[i] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
            if pair_flag == 1:
                width.append(i-pre_loc)
                pair_flag = 0
            else:
                pair_flag = 1
                pre_loc = i
        if ts[i] == thres and (ts[i-1] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
    
    feature_name.extend(['LongThresCrossing_cnt'+'lead'+str(idx), 'LongThresCrossing_width'+str(idx)])
    if len(width) > 1:
        return [cnt, np.mean(width)]
    else:
        return [cnt, 0.0]

def WaveletStat(ts,feature_name,idx):
    '''
    Statistic features for DWT
    '''

    DWTfeat = []
    feature_name.extend(['WaveletStat_'+str(idx)+'lead'+str(i) for i in range(48)])
    if len(ts) >= 1664:
        db7 = pywt.Wavelet('db7')      
        cAcD = pywt.wavedec(ts, db7, level = 7)
        for i in range(8):
            DWTfeat = DWTfeat + [max(cAcD[i]), min(cAcD[i]), np.mean(cAcD[i]),
                                    np.median(cAcD[i]), np.std(cAcD[i])]
            energy = 0
            for j in range(len(cAcD[i])):
                energy = energy + cAcD[i][j] ** 2
            DWTfeat.append(energy/len(ts))
        return DWTfeat
    else:
        return [0.0]*48


def get_long_feature(ts,idx):
    ##feature_list 放的是feature name
    feature_name = []
    ##long_feature放的是对应的特征
    long_feature = []
    long_feature.extend([LongBasicStat(ts,feature_name,idx)])
    long_feature.extend([LongZeroCrossing(ts,0,feature_name,idx)])
    long_feature.extend([LongFFTBandPower(ts,feature_name,idx)])
    long_feature.extend([LongFFTPower(ts,feature_name,idx)])
    long_feature.extend([LongFFTBandPower(ts,feature_name,idx)])
    long_feature.extend([LongSNR(ts,feature_name,idx)])
    long_feature.extend([LongFFTBandPower(ts,feature_name,idx)])
    long_feature.extend([long_autocorr(ts,feature_name,idx)])
    long_feature.extend([long_zigzag(ts,feature_name,idx)])
    long_feature.extend([LongThresCrossing(ts,feature_name,idx)])
    long_feature.extend([WaveletStat(ts,feature_name,idx)])
    long_feature.extend([long_zigzag(ts,feature_name,idx)])
    return feature_name,long_feature