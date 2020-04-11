import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import sys
sys.path.append(r'c:\users\administrator\appdata\local\programs\python\python37\lib\site-packages')
import os
import numpy as np
import biosppy
import wfdb
from pyentrp import entropy as ent
import pywt
from biosppy.signals import ecg

####滤波 使用db6小波对信号进行9级小波分解，去除D1，D2，A9分量，使用剩下的分量进行重构，得到滤波后的信号
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


##### 切片操作
def SegSig_1d(sig, seg_length=1500, overlap_length=0,
          full_seg=True, stt=0):
    """
    # 按指定参数对单导联ECG进行切片
    :param sig: 1-D numpy Array，单导联ECG
    :param seg_length:  int，切片的采样点长度
    :param overlap_length: int, 切片之间相互覆盖的采样点长度，默认为0
    :param full_seg:  bool， 是否对信号末尾不足seg_length的片段进行延拓并切片，默认True
    :param stt:  int, 开始进行切片的位置， 默认从头开始（0）
    :return: 2-D numpy Array, 切片个数 * 切片长度
    """
    length = len(sig)
    SEGs = np.zeros([1, seg_length])
    start = stt
    while start+seg_length <= length:
        tmp = sig[start:start+seg_length].reshape([1, seg_length])
        SEGs = np.concatenate((SEGs, tmp))
        start += seg_length
        start -= overlap_length
    if full_seg:
        if start < length:
            pad_length = seg_length-(length-start)
            tmp = np.concatenate((sig[start:length].reshape([1, length-start]),
                                sig[:pad_length].reshape([1, pad_length])), axis=1)
            SEGs = np.concatenate((SEGs, tmp))
    SEGs = SEGs[1:]
    return SEGs

###将不同长度的样本切片成相同基本片段组成的等长数据
###seg_num 设置为样本最长数据/样本最短数据
####seg_length设置为样本最短数据长度
def Stack_Segs_generate(sig, seg_num=13, seg_length=1500, full_seg=True, stt=0):
    """
    # 对单导联信号滤波，按照指定切片数目和长度进行切片，并堆叠为矩阵
    :param sig: 1-D numpy Array, 输入单导联信号
    :param seg_num: int，指定切片个数
    :param seg_length: int，指定切片采样点长度
    :param full_seg: bool，是否对信号末尾不足seg_length的片段进行延拓并切片，默认True
    :param stt: int, 开始进行切片的位置， 默认从头开始（0）
    :return: 3-D numpy Array, 1 * 切片长度 * 切片个数
    """
    sig = WTfilt_1d(sig)
    if len(sig) < seg_length + seg_num:
        sig = Pad_1d(sig, target_length=(seg_length + seg_num - 1))

    overlap_length = int(seg_length - (len(sig) - seg_length) / (seg_num - 1))

    if (len(sig) - seg_length) % (seg_num - 1) == 0:
        full_seg = False

    SEGs = SegSig_1d(sig, seg_length=seg_length,
                     overlap_length=overlap_length, full_seg=full_seg, stt=stt)
    del sig
    SEGs = SEGs.transpose()
    SEGs = SEGs.reshape([1, SEGs.shape[0], SEGs.shape[1]])
    return SEGs


#####提取HRV特征
####这里的R峰识别方法是hamilton_segmenter，实际使用中可以使用christov_segmenter
class ManFeat_HRV(object):
    """
        针对一条记录的HRV特征提取， 以II导联为基准
    """
    FEAT_DIMENSION = 9

    def __init__(self, sig, fs, min_second):
        assert len(sig.shape) == 1, 'The signal must be 1-dimension.'
        assert sig.shape[0] >= fs * min_second, 'The signal must >= 6 seconds.'
        self.sig = WTfilt_1d(sig)
        self.fs = fs
        self.rpeaks, = ecg.hamilton_segmenter(signal=self.sig, sampling_rate=self.fs)
        self.rpeaks, = ecg.correct_rpeaks(signal=self.sig, rpeaks=self.rpeaks,
                                          sampling_rate=self.fs)
        self.RR_intervals = np.diff(self.rpeaks)
        self.dRR = np.diff(self.RR_intervals)

    def __get_sdnn(self):  # 计算RR间期标准差
        return np.array([np.std(self.RR_intervals)])

    def __get_maxRR(self):  # 计算最大RR间期
        return np.array([np.max(self.RR_intervals)])

    def __get_minRR(self):  # 计算最小RR间期
        return np.array([np.min(self.RR_intervals)])

    def __get_meanRR(self):  # 计算平均RR间期
        return np.array([np.mean(self.RR_intervals)])

    def __get_Rdensity(self):  # 计算R波密度
        return np.array([(self.RR_intervals.shape[0] + 1)
                         / self.sig.shape[0] * self.fs])

    def __get_pNN50(self):  # 计算pNN50
        return np.array([self.dRR[self.dRR >= self.fs * 0.05].shape[0]
                         / self.RR_intervals.shape[0]])

    def __get_RMSSD(self):  # 计算RMSSD
        return np.array([np.sqrt(np.mean(self.dRR * self.dRR))])

    def __get_SampEn(self):  # 计算RR间期采样熵
        sampEn = ent.sample_entropy(self.RR_intervals,
                                    2, 0.2 * np.std(self.RR_intervals))
        for i in range(len(sampEn)):
            if np.isnan(sampEn[i]):
                sampEn[i] = -2
            if np.isinf(sampEn[i]):
                sampEn[i] = -1
        return sampEn

    def extract_features(self):  # 提取HRV所有特征
        features = np.concatenate((self.__get_sdnn(),
                                   self.__get_maxRR(),
                                   self.__get_minRR(),
                                   self.__get_meanRR(),
                                   self.__get_Rdensity(),
                                   self.__get_pNN50(),
                                   self.__get_RMSSD(),
                                   self.__get_SampEn(),
                                   ))
        assert features.shape[0] == ManFeat_HRV.FEAT_DIMENSION
        return features



###统计各样本 长度 返回最长 最短
def sample_length():
    sig_len_all = []
    def readname(filePath):
        name = os.listdir(filePath)
        return name
    file_colletion = readname(sys.path[0])
    dat_collection = []
    for i in range(0,len(file_colletion)):
        if file_colletion[i].find('.mat')>=0:
            dat_collection.append(file_colletion[i].strip('.mat'))
    for j in range(0,len(dat_collection)):
        record = wfdb.rdrecord(dat_collection[j])
        sig_len_all.append(record.__dict__['sig_len']/record.__dict__['fs'])
    return sig_len_all
