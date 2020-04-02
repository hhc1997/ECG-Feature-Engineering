# Feature-Engineering
1.statistic.py 用于统计样本的年龄信息，性别信息，各疾病的数量信息，其中年龄信息和性别信息可作为两个特征，各疾病的数量信息可以分析数据集是否平衡，对某类数量较小的需进行补充（可以直接复制，或者通过VAE等添加人造数据）。

2.resample.py 用于重采样，原始样本数据500HZ频率过高。

3.重采样后的数据请放在 resample_data文件夹里。

4.目前有4部分特征提取，extract_SHORT.py，extract_LONG.py，extract_SHORT.py，extract_QRSF.py，extract_HRV.py，运行后会在FeatureData中生成 4个CSV文件（我已经都生成了，但是有点大上传不了，只上传两个）。

5.特征提取完后运行xgboost_clf.py 打印准确率，加权F1，各类别F1。
