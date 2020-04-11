import sys
sys.path.append(r'c:\users\administrator\appdata\local\programs\python\python37\lib\site-packages')
import matplotlib.pyplot as plt
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
from biosppy.signals import ecg
from utils import gender_num
from utils import label_num
from utils import sample_length
from utils import readname
from utils import flatten
from LONG_features import zigzag
from LONG_features import autocorr
from LONG_features import LongBasicStat
from LONG_features import LongZeroCrossing
from LONG_features import LongFFTBandPower
from LONG_features import LongFFTPower
from LONG_features import LongFFTBandPower
from LONG_features import LongSNR
from LONG_features import long_autocorr
from LONG_features import long_zigzag
from LONG_features import LongThresCrossing
from LONG_features import WaveletStat
from LONG_features import get_long_feature
print('start')



def extract_LONG_in_specific(dat_collection,file_name):

    LONG_feature = pd.DataFrame()
    for j in range(len(dat_collection)):
        record = wfdb.rdrecord('resample_data'+str('/')+dat_collection[j].strip('.mat'))
        data = scio.loadmat('resample_data'+str('/')+dat_collection[j])
        ###以2导联为基准
        if  ',' not in record.__dict__['comments'][2][4:]:
            LONG_feature_each_lead = pd.DataFrame(data = [flatten(get_long_feature(data['val'][0],0)[1])] ,columns = get_long_feature(data['val'][0],0)[0])
            for e in range(1,len(data['val'])):
                ecg_lead = data['val'][e]
                name_and_value = get_long_feature(ecg_lead,e)
                temp_LONG_feature = pd.DataFrame(data = [flatten(name_and_value[1])] ,columns = name_and_value[0])
                LONG_feature_each_lead  = LONG_feature_each_lead .join(temp_LONG_feature)
            LONG_feature = LONG_feature.append(LONG_feature_each_lead)
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
                LONG_feature_each_lead = pd.DataFrame(data = [flatten(get_long_feature(data['val'][0],0)[1])] ,columns = get_long_feature(data['val'][0],0)[0])
                for e in range(1,len(data['val'])):
                    ecg_lead = data['val'][e]
                    name_and_value = get_long_feature(ecg_lead,e)
                    temp_LONG_feature = pd.DataFrame(data = [flatten(name_and_value[1])] ,columns = name_and_value[0])
                    LONG_feature_each_lead  = LONG_feature_each_lead .join(temp_LONG_feature)
                LONG_feature = LONG_feature.append(LONG_feature_each_lead)
    
        print('finish'+str(j))
    LONG_feature.to_csv(str(file_name)+'_Features/'+'LONG_FEATURE.csv', sep=',', header=True, index=False)
