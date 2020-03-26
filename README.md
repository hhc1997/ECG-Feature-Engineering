# Feature-Engineering
1.statistic.py 用于统计样本的年龄信息，性别信息，各疾病的数量信息，其中年龄信息和性别信息可作为两个特征，各疾病的数量信息可以分析数据集是否平衡，对某类数量较小的需进行补充（可以直接复制，或者通过VAE等添加人造数据）。
2.resample.py 用于重采样，原始样本数据500HZ频率过高。

3.extract_feature.py 用于统一各样本的长度，以及提取HRV特征。