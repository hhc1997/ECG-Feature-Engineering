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
import wfdb
from pyentrp import entropy as ent
import pywt
from biosppy.signals import ecg
from utils import gender_num
from utils import label_num
from utils import sample_length
from utils import readname
from HRV_features import Stack_Segs_generate
from HRV_features import ManFeat_HRV
print('start')

####为了与神经网络模型的训练集 测试集一致 dat_collection是划分的数据名集合，file_name 是['train','test']二选一
def extract_HRV_in_specific(dat_collection,file_name):
    minimum_second = int(min(sample_length()))
    HRV_feature = pd.DataFrame(columns=['label','sdnn','maxRR','minRR','meanRR','Rdensity','pNN50','RMSSD','SampEn1','SampEn2','gender','age'])
    for j in range(0,len(dat_collection)):
        
        record = wfdb.rdrecord('resample_data'+str('/')+dat_collection[j].strip('.mat'))
        data = scio.loadmat('resample_data'+str('/')+dat_collection[j])
        print('Data read:'+time.strftime("%H:%M:%S"))
        ###以2导联为基准
        ecg_lead_2 = data['val'][1]
        Feat_HRV = ManFeat_HRV(ecg_lead_2, 300, minimum_second)
        lead_2_HRV = Feat_HRV.extract_features()
        print('Feature extracted:'+time.strftime("%H:%M:%S"))
        if  ',' not in record.__dict__['comments'][2][4:]:
            HRV_feature  = HRV_feature .append([{'label':label_num(record.__dict__['comments'][2][4:]),
                                                'sdnn':lead_2_HRV[0],
                                                'maxRR':lead_2_HRV[1],
                                                'minRR':lead_2_HRV[2],
                                                'meanRR':lead_2_HRV[3],
                                                'Rdensity':lead_2_HRV[4],
                                                'pNN50':lead_2_HRV[5],
                                                'RMSSD':lead_2_HRV[6],
                                                'SampEn1':lead_2_HRV[7],
                                                'SampEn2': lead_2_HRV[8],
                                                'gender':gender_num(record.__dict__['comments'][1][5:]),
                                                'age':record.__dict__['comments'][0][5:]}],
                                                ignore_index=True)
            print('Feature_load1:' + time.strftime("%H:%M:%S"))
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
                HRV_feature = HRV_feature.append([{'label': label_num(ans),
                                                'sdnn': lead_2_HRV[0],
                                                'maxRR': lead_2_HRV[1],
                                                'minRR': lead_2_HRV[2],
                                                'meanRR': lead_2_HRV[3],
                                                'Rdensity': lead_2_HRV[4],
                                                'pNN50': lead_2_HRV[5],
                                                'RMSSD': lead_2_HRV[6],
                                                'SampEn1': lead_2_HRV[7],
                                                'SampEn2': lead_2_HRV[8],
                                                'gender': gender_num(record.__dict__['comments'][1][5:]),
                                                'age': record.__dict__['comments'][0][5:]}],
                                                    ignore_index=True)
            print('Feature_load2:' + time.strftime("%H:%M:%S"))
        print('finish'+str(j))
    HRV_feature.to_csv(str(file_name)+'_Features/'+'HRV_FEATURE.csv', sep=',', header=True, index=False)

