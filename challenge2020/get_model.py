import sys
sys.path.append(r'c:\users\administrator\appdata\local\programs\python\python37\lib\site-packages')
from models.inceptiontime import Classifier_INCEPTION
from keras.callbacks import TensorBoard
import tensorflow.keras as keras
import wfdb
import pandas as pd
import scipy.io as scio
import os
from utils.loss import my_loss, f1
model = Classifier_INCEPTION(bottleneck_size=32, depth=6, kernel_size=40, nb_filters=32, type='inceptiontime_v2',head_dropout_rate=0.5).model
adam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam,
              loss= 'categorical_crossentropy',
              metrics=['accuracy',f1])
model.summary()
model.load_weights('./result/result-inceptiontime/best_model.hdf5')

def readname(filePath):
    name = os.listdir(filePath)
    return name
file_colletion = readname('../resample_data'+str('/'))
dat_collection = []
for i in range(0,len(file_colletion)):
    if file_colletion[i].find('.mat')>=0:
        dat_collection.append(file_colletion[i].strip('.mat'))
deep_feature = pd.DataFrame(columns=['D1','D2','D3','D4','D5','D6','D7','D8','D9'])
dat_collection.sort()
for j in range(0,len(dat_collection)):
    print(dat_collection[j])
    record = wfdb.rdrecord('../resample_data'+str('/')+dat_collection[j].strip('.mat'))
    data_temp = pd.DataFrame(columns=['D1','D2','D3','D4','D5','D6','D7','D8','D9'],data = model.predict(scio.loadmat('/data/hanhaochen/fixed_length_300HZ'+str('/')+dat_collection[j])['val'].T.reshape(1,43200,12)).tolist())
    if  ',' not in record.__dict__['comments'][2][4:]:
        deep_feature = deep_feature.append(data_temp)
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
            deep_feature = deep_feature.append(data_temp)
deep_feature.to_csv('DEEP_FEATURE.csv', sep=',', header=True, index=False)
