import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
data = pd.read_csv('FeatureData\HRV_FEATURE.csv', encoding = 'UTF-8')
data1 = pd.read_csv('FeatureData\LONG_FEATURE.csv', encoding = 'UTF-8')
data2 = pd.read_csv('FeatureData\SHORT_FEATURE.csv', encoding = 'UTF-8')
data3 = pd.read_csv('FeatureData\QRS_FEATURE.csv', encoding = 'UTF-8')
data = data.join(data1)
data = data.join(data2)
data = data.join(data3)
# data 数据操作

X = np.array(data)[:,1:] # (2189, 32)
Y = np.array(data)[:,0]
X_train, X_validation, y_train, y_validation = train_test_split(X,Y,test_size=0.3,random_state=123)
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
XGC = XGBClassifier(max_depth=6,
                     learning_rate=0.1,
                      n_estimators=100,
                      silent=True
                     )
eval_set = [(X_validation, y_validation)]
XGC.fit(X_train, y_train,verbose=True,eval_set = eval_set)
# make predictions for test data
y_pred = XGC.predict(X_validation)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_validation, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

print('加权F1:' + str(f1_score(y_validation, y_pred, average='weighted')))
# 总共输出4个数组，依次是precision、recall、fscore、support
print('各标签F1：' + str(precision_recall_fscore_support(y_validation, y_pred, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9])[2]))


####特征选择 选择重要性排序前300的
fea_ = XGC.feature_importances_
fea_name = data.columns.values.tolist()[1:]
imp_fea_300 = np.argsort(-fea_ )[:300]
###第一列是标签 所以要加1
imp_fea_300 = [i + 1 for i in imp_fea_300]
imp_fea_300.insert(0,0)
data_fea_select = data.iloc[0:,imp_fea_300]
