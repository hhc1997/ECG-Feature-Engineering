import pandas as pd
import numpy as np
import scipy.io as scio
import sys
import os
import wfdb
from scipy import signal
import matplotlib.pyplot as plt

def readname(filePath):
    name = os.listdir(filePath)
    return name
file_colletion = readname(sys.path[0])
dat_collection = []
for i in range(0,len(file_colletion)):
    if file_colletion[i].find('.mat')>=0:
        dat_collection.append(file_colletion[i].strip('.mat'))
patient_info = pd.DataFrame(columns = ['Age','Sex','AF','I-AVB','LBBB','Normal','PAC','PVC','RBBB','STD','STE'])
for j in range(0,len(dat_collection)):
    record = wfdb.rdrecord(dat_collection[j])
    patient_info = patient_info.append([{'Age':record.__dict__['comments'][0][5:],'Sex':record.__dict__['comments'][1][5:],'AF':0,'I-AVB':0,'LBBB':0,'Normal':0,'PAC':0,'PVC':0,'RBBB':0,'STD':0,'STE':0}], ignore_index=True)
    if  ',' not in record.__dict__['comments'][2][4:]:
        patient_info.loc[j][record.__dict__['comments'][2][4:]] = 1
    else:
        s = record.__dict__['comments'][2][4:]
        s += '0'
        while len(s) != 0:
            ans = ''
            for k in range(len(s)):
                if s[k] != ',' and s[k] != '0':
                    ans+=s[k]
                elif s[k] == ',':
                    s = s[k+1:]
                    break
                elif s[k]== '0':
                    s = ''
            patient_info.loc[j][ans] = 1
patient_info.to_csv('info.csv', sep=',', header=True, index=False)