import scipy.io
import numpy as np
import glob
from scipy.signal import firwin, lfilter, spectrogram
import math
from biosppy.signals import ecg as ecgprocess
import neurokit as nk
import gc
import os
from collections import Counter
import pickle as dill

import seaborn as sns
import matplotlib.pyplot as plt
dataDir = '/data/weiyuhua/data/Challenge2018_300hz/'
# dataDir = '../raw_data/training2017/'
from sklearn.externals import joblib
FS = 300

TIME_OF_WINDOW = 60
OVERLAP = 0.5
WINDOW_SIZE = 18000

# R peak detector
# def detect_beats(
#         ecg,  # The raw ECG signal
#         rate,  # Sampling rate in HZ
#         # Window size in seconds to use for
#         ransac_window_size=5.0,
#         # Low frequency of the band pass filter
#         lowfreq=5.0,
#         # High frequency of the band pass filter
#         highfreq=15.0,
# ):
#     """
#     ECG heart beat detection based on
#     http://link.springer.com/article/10.1007/s13239-011-0065-3/fulltext.html
#     with some tweaks (mainly robust estimation of the rectified signal
#     cutoff threshold).
#     """
#
#     ransac_window_size = int(ransac_window_size * rate)
#
#     lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
#     highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
#     # TODO: Could use an actual bandpass filter
#     ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
#     ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
#
#     # Square (=signal power) of the first difference of the signal
#     decg = np.diff(ecg_band)
#     decg_power = decg ** 2
#
#     # Robust threshold and normalizator estimation
#     thresholds = []
#     max_powers = []
#     for i in range(int(len(decg_power) / ransac_window_size)):
#         sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
#         d = decg_power[sample]
#         thresholds.append(0.5 * np.std(d))
#         max_powers.append(np.max(d))
#
#     threshold = np.median(thresholds)
#     max_power = np.median(max_powers)
#     decg_power[decg_power < threshold] = 0
#
#     decg_power /= max_power
#     decg_power[decg_power > 1.0] = 1.0
#     square_decg_power = decg_power ** 2
#
#     shannon_energy = -square_decg_power * np.log(square_decg_power)
#     shannon_energy[~np.isfinite(shannon_energy)] = 0.0
#
#     mean_window_len = int(rate * 0.125 + 1)
#     lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
#     # lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)
#
#     lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 8.0)
#     lp_energy_diff = np.diff(lp_energy)
#
#     zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
#     zero_crossings = np.flatnonzero(zero_crossings)
#     zero_crossings -= 1
#     return zero_crossings

## Loading time serie signals

files = sorted(glob.glob(dataDir + "*.mat"))
ori_trainset = []
size = 6877-34
length = []
count = 0
index2del = [1216, 1670, 2013, 2021, 2214, 2478, 2659, 2878, 3230, 5904, 6674,
             592, 649, 863, 1389, 1580, 2006, 2799, 3203, 3497, 3842, 4004,
             4080, 4133, 4652, 4787, 4841, 5189, 5618, 5828, 5841, 6230, 6696, 6850]
for i in range(len(files)):
    if i+1 in index2del:
        continue
    f = files[i]
    record = f[:-4]
    record = record[-6:]
    # Loading
    mat_data = scipy.io.loadmat(f[:-4] + ".mat")
    print('Loading record {}'.format(record))
    data = mat_data['val'].squeeze()
    # Preprocessing
    print('Preprocessing record {}'.format(record))
    data = np.nan_to_num(data)  # removing NaNs and Infs
    #data = data - np.mean(data)
    #data = data / np.std(data)
    # data = fir(data)
    ori_trainset.append(data)

    length.append(data.shape[1])
    # if length[i] > 30000:
    #     print("**************************************************")
    #     print(length[i])
    #     print("**************************************************")
    # print(1)
# sns.distplot(length, hist=True, rug=False) # rug=True表示显示 rugplot(x)
# plt.show()

length = np.array(length)
print(np.max(length))
# max length = 43200
## Loading labels
import csv

csvfile = list(csv.reader(open(dataDir + 'REFERENCE.csv')))
csvfile = csvfile[1:]
ori_traintarget = np.zeros((len(ori_trainset), 9))
classes = ['1', '2', '3', '4','5', '6', '7', '8','9']
# 1: 918   Normal                                        0  1
# 2: 1098  Atrial fibrillation (AF)                      1  1
# 3: 704   First-degree atrioventricular block (I-AVB)   2  1.2
# 4: 207   Left bundle branch block (LBBB)               3  5
# 5: 1695  Right bundle branch block (RBBB)              4  0.7
# 6: 574   Premature atrial contraction (PAC)            5  2
# 7: 653   Premature ventricular contraction (PVC)       6  1.5
# 8: 826   ST-segment depression (STD)                   7  1.2
# 9: 202   ST-segment elevated (STE)                     8  5
csvfile2 = []
for i in range(len(csvfile)):
    if i+1 in index2del:
        continue
    csvfile2.append(csvfile[i])

label = []
for row in range(len(csvfile2)):
    label.append(csvfile2[row][1])
    ori_traintarget[row, classes.index(csvfile2[row][1])] = 1
label = np.array(label)

print(Counter(label))

ori_trainset_tmp = []
ori_trainset_II = []
ori_traintarget_tmp = []
for i in range(len(ori_traintarget)):
    ori_trainset_II.append(ori_trainset[i][1])
    mask = np.zeros((12, 43200))
    a = ori_trainset[i]
    # # 最长30000
    # if a.shape[1]>30000:
    #     mask = a[:, :30000]
    # else:
    mask[:, :a.shape[1]] = a
    ori_trainset_tmp.append(mask)
    ori_traintarget_tmp.append(ori_traintarget[i])
ori_trainset = ori_trainset_tmp
ori_traintarget = ori_traintarget_tmp


ratio_train = 0.8
N = len(ori_trainset)
ind_cut = int(ratio_train*N)
ind_cut2 = int((ratio_train + 0.1)*N)
ind = np.random.permutation(N)
ori_trainset = np.array(ori_trainset)
ori_trainset_II = np.array(ori_trainset_II)
ori_traintarget = np.array(ori_traintarget)

x_train, x_val, x_test = ori_trainset[ind[0:ind_cut]], ori_trainset[ind[ind_cut:ind_cut2]], ori_trainset[ind[ind_cut2:N]]
y_train, y_val, y_test = ori_traintarget[ind[0:ind_cut]], ori_traintarget[ind[ind_cut:ind_cut2]], ori_traintarget[ind[ind_cut2:N]]
length_train, length_val, length_test = length[ind[0:ind_cut]], length[ind[ind_cut:ind_cut2]], length[ind[ind_cut2:N]]
x_train_II, x_val_II, x_test_II = ori_trainset_II[ind[0:ind_cut]], ori_trainset_II[ind[ind_cut:ind_cut2]], ori_trainset_II[ind[ind_cut2:N]]

errors_PeakSet = np.zeros((6877,))
errors_SpectSet = np.zeros((6877,))
def getRpeakSet(ecgset, mode='0-1', index2del=[]):
    rpeakSet = np.zeros((len(ecgset), 18000))
    pwaveSet = np.zeros((len(ecgset), 18000))
    qwaveSet = np.zeros((len(ecgset), 18000))
    twaveSet = np.zeros((len(ecgset), 18000))
    for i in range(len(ecgset)):
      #  if i== 1215: 1979
      #      continue
        ecg = ecgset[i]
        # rpeaks = ecgprocess.gamboa_segmenter(ecg[0:30000], 300)
        print(i)
        print(len(ecgset))
        if i in index2del:
            continue
        try:
            processed_ecg = nk.ecg_process(ecg[0:18000], sampling_rate=300)
        except Exception as e:
            print(e)
            errors_PeakSet[i] = 1
            continue
        rpeaks = processed_ecg['ECG']['R_Peaks']
        # rpeaks = detect_beats(ecg[0:30000], 300)
        for r in rpeaks:
            if (r > len(rpeakSet[i])):
                print(r)
            else:
                if mode == '0-1':
                    rpeakSet[i][r] = 1
                else:
                    rpeakSet[i][r] = 1000
        pwaves = processed_ecg['ECG']['P_Waves']
        # rpeaks = detect_beats(ecg[0:30000], 300)
        for r in pwaves:
            if (r > len(pwaveSet[i])):
                print(r)
            else:
                if mode == '0-1':
                    pwaveSet[i][r] = 1
                else:
                    pwaveSet[i][r] = 1000
        qwaves = processed_ecg['ECG']['Q_Waves']
        # rpeaks = detect_beats(ecg[0:30000], 300)
        for r in qwaves:
            if (r > len(qwaveSet[i])):
                print(r)
            else:
                if mode == '0-1':
                    qwaveSet[i][r] = 1
                else:
                    qwaveSet[i][r] = 1000
        twaves = processed_ecg['ECG']['T_Waves']
        # rpeaks = detect_beats(ecg[0:30000], 300)
        for r in twaves:
            if (r > len(twaveSet[i])):
                print(r)
            else:
                if mode == '0-1':
                    twaveSet[i][r] = 1
                else:
                    twaveSet[i][r] = 1000
        del rpeaks, pwaves, qwaves, twaves, r, ecg
        gc.collect()

    return np.array([np.array(rpeakSet), np.array(pwaveSet), np.array(qwaveSet), np.array(twaveSet)])
def getSpecSet(ecgset):
    specSet = np.zeros((len(ecgset), 33, 300))
    for i in range(len(ecgset)):
        ecg = ecgset[i]
        f, t, spec = spectrogram(ecg, 300, nperseg=64, noverlap=0.5)
        spec = np.log(spec)
        for j in range(33):
            for k in range(len(spec[j])):
                try:
                    specSet[i][j][k] = spec[j][k]
                except Exception as e:
                    errors_SpectSet[i] = 1
                    print(e)
                    continue
        # specSet[i][:][0:len(spec[0])] = spec
    return specSet

peakset = getRpeakSet(ori_trainset_II[0:len(ori_trainset_II)], mode='0-1')

errors_PeakSet_index =[]
for i in range(len(errors_PeakSet)):
    if errors_PeakSet[i] == 1:
        errors_PeakSet_index.append(i+1)
print(errors_PeakSet_index)
# 500hz[1216, 1670, 2013, 2021, 2214, 2478, 2659, 2878, 3230, 5904, 6674]
# 300hz[1216, 1670, 2013, 2021, 2214, 2478, 2659, 2878, 3230, 5904, 6674]

Specset = getSpecSet(ori_trainset_II)
errors_SpectSet_index =[]
for i in range(len(errors_SpectSet)):
    if errors_SpectSet[i] == 1:
        errors_SpectSet_index.append(i+1)
print(errors_SpectSet_index)
# 500hz[592, 649, 863, 1389, 1580, 1670, 2006, 2799, 3203, 3497, 3842, 4004, 4080, 4133, 4652, 4787, 4841, 5189, 5618, 5828, 5841, 6230, 6696, 6850]
# 300hz[592, 649, 863, 1389, 1580, 1670, 2006, 2799, 3203, 3497, 3842, 4004, 4080, 4133, 4652, 4787, 4841, 5189, 5618, 5828, 5841, 6230, 6696, 6850]

peakset_train, peakset_val, peakset_test = [peakset[0][ind[0:ind_cut]], peakset[1][ind[0:ind_cut]], peakset[2][ind[0:ind_cut]], peakset[3][ind[0:ind_cut]]],\
                                           [peakset[0][ind[ind_cut:ind_cut2]], peakset[1][ind[ind_cut:ind_cut2]],peakset[2][ind[ind_cut:ind_cut2]],peakset[3][ind[ind_cut:ind_cut2]]],\
                                           [peakset[0][ind[ind_cut2:N]], peakset[1][ind[ind_cut2:N]], peakset[2][ind[ind_cut2:N]], peakset[3][ind[ind_cut2:N]]]

trainsetSpec = getSpecSet(x_train_II)
valsetSpec = getSpecSet(x_val_II)
testsetSpec = getSpecSet(x_test_II)

### preprocess training data ###
trainset = []
traintarget = []
trainSpec = []
trainPeaks = [[],[],[],[]]
max = 0

# 1: 918   Normal                                        0  1
# 2: 1098  Atrial fibrillation (AF)                      1  1
# 3: 704   First-degree atrioventricular block (I-AVB)   2  1.2
# 4: 207   Left bundle branch block (LBBB)               3  5
# 5: 1695  Right bundle branch block (RBBB)              4  0.7
# 6: 574   Premature atrial contraction (PAC)            5  2
# 7: 653   Premature ventricular contraction (PVC)       6  1.5
# 8: 826   ST-segment depression (STD)                   7  1.2
# 9: 202   ST-segment elevated (STE)                     8  5

for i in range(len(x_train)):
    np.random.seed(0)
    aug_fact = 1
    if y_train[i][2] == 1:
        # aug_fact = 1.2
        p = np.array([0.8, 0.2])
        aug_fact = np.random.choice([1, 2], p=p.ravel())
    if y_train[i][3] == 1:
        aug_fact = 5
    if y_train[i][4] == 1:
        # aug_fact = 0.7
        p = np.array([0.7, 0.3])
        aug_fact = np.random.choice([1, 0], p=p.ravel())
    if y_train[i][5] == 1:
        aug_fact = 2
    if y_train[i][6] == 1:
        # aug_fact = 1.5
        p = np.array([0.5, 0.5])
        aug_fact = np.random.choice([1, 2], p=p.ravel())
    if y_train[i][7] == 1:
        # aug_fact = 1.2
        p = np.array([0.8, 0.2])
        aug_fact = np.random.choice([1, 2], p=p.ravel())
    if y_train[i][8] == 1:
        aug_fact = 5

    for j in range(aug_fact):

        offset = 0
        len_train = int(length_train[i])
        while offset == 0 or offset + WINDOW_SIZE < len_train:
            slice = np.zeros((12, WINDOW_SIZE))
            if offset + WINDOW_SIZE < len_train:
                slice[:,:] = x_train[i,:, offset: offset + WINDOW_SIZE]
            else:
                slice[:,:len_train] = x_train[i,:,:len_train]
            trainset.append(slice)
            traintarget.append(y_train[i])
            trainSpec.append(trainsetSpec[i])
            trainPeaks[0].append(peakset_train[0][i])
            trainPeaks[1].append(peakset_train[1][i])
            trainPeaks[2].append(peakset_train[2][i])
            trainPeaks[3].append(peakset_train[3][i])
            offset += int(WINDOW_SIZE * (1 - OVERLAP))


### preprocess training data END ###

### preprocess validation data ###
val_set = []
val_target = []
ecg_val_index = []
for i in range(len(x_val)):
    offset = 0
    count = 0
    len_val = int(length_val[i])
    while offset == 0 or offset + WINDOW_SIZE < len_val:
        slice = np.zeros((12,WINDOW_SIZE))
        if offset + WINDOW_SIZE < len_val:
            slice[:, :] = x_val[i, :, offset: offset + WINDOW_SIZE]
        else:
            slice[:, :len_val] = x_val[i, :, :len_val]
        val_set.append(slice)
        val_target.append(y_val[i])
        offset += int(WINDOW_SIZE * (1 - OVERLAP))
        count += 1
    ecg_val_index.append(count)
### preprocess validation data END ###

### preprocess test data ###
final_testset = []
final_testtarget = []
ecg_test_index = []
for i in range(len(x_test)):
    offset = 0
    count = 0
    len_val = int(length_test[i])
    while offset == 0 or offset + WINDOW_SIZE < len_val:
        slice = np.zeros((12, WINDOW_SIZE))
        if offset + WINDOW_SIZE < len_val:
            slice[:, :] = x_test[i, :, offset: offset + WINDOW_SIZE]
        else:
            slice[:, :len_val] = x_test[i, :, :len_val]
        final_testset.append(slice)
        final_testtarget.append(y_test[i])
        offset += int(WINDOW_SIZE * (1 - OVERLAP))
        count += 1
    ecg_test_index.append(count)
### preprocess test data END ###

def scale_input(X):
    items = []
    for x in X:
        item = []
        sum = 0
        count = 0
        for a in x:
            if(a != 0.0):
                sum += a
                count += 1
        mean_x = sum / count
        print(mean_x)
        for a in x:
            if(a != mean_x):
                a = (a - mean_x) / abs(a - mean_x) * math.log10(abs(a - mean_x) + 1) + mean_x
            item.append(a)
        items.append(item)
    return items

trainset = np.array(trainset)
traintarget = np.array(traintarget)
trainSpec = np.array(trainSpec)
trainPeaks = [np.array(trainPeaks[0]), np.array(trainPeaks[1]), np.array(trainPeaks[2]), np.array(trainPeaks[3])]

val_set = np.array(val_set)
val_target = np.array(val_target)

final_testset = np.array(final_testset)
final_testtarget = np.array(final_testtarget)

ind_train = np.random.permutation(len(trainset))
trainset = trainset[ind_train]
traintarget = traintarget[ind_train]
trainsetSpec = trainSpec[ind_train]
trainsetPeaks = np.array([trainPeaks[0][ind_train], trainPeaks[1][ind_train], trainPeaks[2][ind_train], trainPeaks[3][ind_train]])


data_path = '/data/weiyuhua/data/Challenge2018_300hz/preprocessed_data_new/'

res = {'trainset':trainset, 'traintarget':traintarget,'trainsetSpec':trainsetSpec}
with open(os.path.join(data_path, 'data_aug_train.pkl'), 'wb') as fout:
    dill.dump(res, fout,protocol=4)

res = {'val_set': val_set, 'val_target': val_target, 'ecg_val_index': ecg_val_index, 'valsetSpec': valsetSpec, 'peakset_val': peakset_val}
with open(os.path.join(data_path, 'data_aug_val.pkl'), 'wb') as fout:
    dill.dump(res, fout,protocol=4)

res = {'final_testset': final_testset, 'final_testtarget': final_testtarget, 'ecg_test_index': ecg_test_index, 'testsetSpec': testsetSpec, 'peakset_test': peakset_test}
with open(os.path.join(data_path, 'data_aug_test.pkl'), 'wb') as fout:
    dill.dump(res, fout,protocol=4)

res = {'trainsetPeaks-r': trainsetPeaks[0], 'valsetPeaks-r': peakset_val[0], 'testsetPeaks-r': peakset_test[0]}
with open(os.path.join(data_path, 'data_aug-rwave.pkl'), 'wb') as fout:
    dill.dump(res, fout,protocol=4)

res = {'trainsetPeaks-p': trainsetPeaks[1], 'valsetPeaks-p': peakset_val[1], 'testsetPeaks-p': peakset_test[1]}
with open(os.path.join(data_path, 'data_aug-pwave.pkl'), 'wb') as fout:
    dill.dump(res, fout,protocol=4)

res = {'trainsetPeaks-q': trainsetPeaks[2], 'valsetPeaks-q': peakset_val[2], 'testsetPeaks-q': peakset_test[2]}
with open(os.path.join(data_path, 'data_aug-qwave.pkl'), 'wb') as fout:
    dill.dump(res, fout,protocol=4)

res = {'trainsetPeaks-t': trainsetPeaks[3], 'valsetPeaks-t': peakset_val[3], 'testsetPeaks-t': peakset_test[3]}
with open(os.path.join(data_path, 'data_aug-twave.pkl'), 'wb') as fout:
    dill.dump(res, fout,protocol=4)

# Saving both
# l = len(trainset)
# half_l = int(l/2)
# quarter_l = int(half_l/2)

# n = int(l/48)
# for i in range(48):
#     start = i * n
#     if i == 47:
#         end = l
#     else:
#         end = (i + 1) * n
#     scipy.io.savemat('/data/weiyuhua/data/Challenge2018/preprocessed_data_new/data_aug_train_%d.mat' %(i+1),
#                      mdict={'trainset': trainset[start:end],'traintarget': traintarget[start:end], 'trainsetSpec': trainsetSpec[start:end]})
#
# l1 = len(trainPeaks[0])
# n = int(l/2)
# for i in range(2):
#
#     scipy.io.savemat('/data/weiyuhua/data/Challenge2018/preprocessed_data_new/data_aug-rwave.mat',
#                      mdict={'trainsetPeaks-r': trainsetPeaks[0], 'valsetPeaks-r': peakset_val[0], 'testsetPeaks-r': peakset_test[0]})
#     scipy.io.savemat('/data/weiyuhua/data/Challenge2018/preprocessed_data_new/data_aug-pwave.mat',
#                      mdict={'trainsetPeaks-p': trainsetPeaks[1], 'valsetPeaks-p': peakset_val[1], 'testsetPeaks-p': peakset_test[1]})
#     scipy.io.savemat('/data/weiyuhua/data/Challenge2018/preprocessed_data_new/data_aug-qwave.mat',
#                      mdict={'trainsetPeaks-q': trainsetPeaks[2], 'valsetPeaks-q': peakset_val[2], 'testsetPeaks-q': peakset_test[2]})
#     scipy.io.savemat('/data/weiyuhua/data/Challenge2018/preprocessed_data_new/data_aug-twave.mat',
#                      mdict={'trainsetPeaks-t': trainsetPeaks[3], 'valsetPeaks-t': peakset_val[3], 'testsetPeaks-t': peakset_test[3]})
#
# l = len(val_set)
# n = int(l/24)
# # ../ preprocessed_data_new /
# for i in range(24):
#     start = i * n
#     if i == 23:
#         end = l
#     else:
#         end = (i + 1) * n
#     scipy.io.savemat('/data/weiyuhua/data/Challenge2018/preprocessed_data_new/data_aug_val_%d.mat' %(i+1),
#                  mdict={'val_set': val_set[start:end], 'val_target': val_target[start:end], 'ecg_val_index': ecg_val_index[start:end], 'valsetSpec': valsetSpec[start:end], 'peakset_val': peakset_val[start:end]})
# scipy.io.savemat('/data/weiyuhua/data/Challenge2018/preprocessed_data_new/data_aug_test.mat',
#                  mdict={'final_testset': final_testset, 'final_testtarget': final_testtarget, 'ecg_test_index': ecg_test_index, 'testsetSpec': testsetSpec, 'peakset_test': peakset_test})
