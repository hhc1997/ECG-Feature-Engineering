import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
data = pd.read_csv('HRV_FEATURE.csv', encoding = 'UTF-8')
data1 = pd.read_csv('LONG_FEATURE.csv', encoding = 'UTF-8')
data2 = pd.read_csv('SHORT_FEATURE.csv', encoding = 'UTF-8')
data3 = pd.read_csv('QRS_FEATURE.csv', encoding = 'UTF-8')
data4 = pd.read_csv('DEEP_FEATURE.csv', encoding = 'UTF-8')
data = data.join(data1)
data = data.join(data2)
data = data.join(data3)
data = data.join(data4)
# data 数据操作

X = np.array(data)[:,1:] # (2189, 32)
Y = np.array(data)[:,0]
X_train, X_validation, y_train, y_validation = train_test_split(X,Y,test_size=0.3,random_state=123)
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
def labels_onehot(label):
    label_onehot = [0]*9
    label_onehot[int(label)-1] = 1
    return label_onehot
##################################
def compute_beta_score(labels, output, beta, num_classes, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(output) != len(labels):
            raise Exception('Numbers of outputs and labels must be the same.')

    # Populate contingency table.
    num_recordings = len(labels)

    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    # Weight function
    C_l = np.ones(num_classes);

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):

            num_labels = np.sum(labels[i])

            if labels[i][j] and output[i][j]:
                tp += 1 / num_labels
            elif not labels[i][j] and output[i][j]:
                fp += 1 / num_labels
            elif labels[i][j] and not output[i][j]:
                fn += 1 / num_labels
            elif not labels[i][j] and not output[i][j]:
                tn += 1 / num_labels

        # Summarize contingency table.
        if ((1 + beta ** 2) * tp + (fn * beta ** 2) + fp):
            fbeta_l[j] = float((1 + beta ** 2) * tp) / float(((1 + beta ** 2) * tp) + (fn * beta ** 2) + fp)
        else:
            fbeta_l[j] = 1.0

        if (tp + fp + beta * fn):
            gbeta_l[j] = float(tp) / float(tp + fp + beta * fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0

    for i in range(num_classes):
        f_beta += fbeta_l[i] * C_l[i]
        g_beta += gbeta_l[i] * C_l[i]
        f_measure += fmeasure_l[i] * C_l[i]
        accuracy += accuracy_l[i] * C_l[i]

    f_beta = float(f_beta) / float(num_classes)
    g_beta = float(g_beta) / float(num_classes)
    f_measure = float(f_measure) / float(num_classes)
    accuracy = float(accuracy) / float(num_classes)

    return accuracy, f_measure, f_beta, g_beta
#####################################################################
def compute_auc(labels, probabilities, num_classes, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(labels) != len(probabilities):
            raise Exception('Numbers of outputs and labels must be the same.')

    find_NaNs = np.isnan(probabilities);
    probabilities[find_NaNs] = 0;

    auroc_l = np.zeros(num_classes)
    auprc_l = np.zeros(num_classes)

    auroc = 0
    auprc = 0

    # Weight function - this will change
    C_l = np.ones(num_classes);

    # Populate contingency table.
    num_recordings = len(labels)

    for k in range(num_classes):

        # Find probabilities thresholds.
        thresholds = np.unique(probabilities[:, k])[::-1]
        if thresholds[0] != 1:
            thresholds = np.insert(thresholds, 0, 1)
        if thresholds[-1] == 0:
            thresholds = thresholds[:-1]

        m = len(thresholds)

        # Populate contingency table across probabilities thresholds.
        tp = np.zeros(m)
        fp = np.zeros(m)
        fn = np.zeros(m)
        tn = np.zeros(m)

        # Find indices that sort the predicted probabilities from largest to
        # smallest.
        idx = np.argsort(probabilities[:, k])[::-1]

        i = 0
        for j in range(m):
            # Initialize contingency table for j-th probabilities threshold.
            if j == 0:
                tp[j] = 0
                fp[j] = 0
                fn[j] = np.sum(labels[:, k])
                tn[j] = num_recordings - fn[j]
            else:
                tp[j] = tp[j - 1]
                fp[j] = fp[j - 1]
                fn[j] = fn[j - 1]
                tn[j] = tn[j - 1]
            # Update contingency table for i-th largest predicted probability.
            while i < num_recordings and probabilities[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize contingency table.
        tpr = np.zeros(m)
        tnr = np.zeros(m)
        ppv = np.zeros(m)
        npv = np.zeros(m)

        for j in range(m):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = 1
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = 1
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = 1
            if fn[j] + tn[j]:
                npv[j] = float(tn[j]) / float(fn[j] + tn[j])
            else:
                npv[j] = 1

        # Compute AUROC as the area under a piecewise linear function with TPR /
        # sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR / recall (x-axis) and PPV / precision
        # (y-axis).

        for j in range(m - 1):
            auroc_l[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc_l[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    for i in range(num_classes):
        auroc += auroc_l[i] * C_l[i]
        auprc += auprc_l[i] * C_l[i]

    auroc = float(auroc) / float(num_classes)
    auprc = float(auprc) / float(num_classes)

    return auroc, auprc
###########################################################################
print('100 hahah')
XGC = XGBClassifier(max_depth=8,
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

label_oh = [labels_onehot(i) for i in y_validation]
print('accuracy, f_measure, f_beta, g_beta:')
print(compute_beta_score(np.array([labels_onehot(i) for i in y_validation]) ,[labels_onehot(i) for i in y_pred],2,9))
print('auroc, auprc')
print(compute_auc(np.array([labels_onehot(i) for i in y_validation]),np.array(XGC.predict_proba(X_validation)),2,9))




#########特征选择
fea_ = XGC.feature_importances_
fea_name = data.columns.values.tolist()[1:]
imp_fea_300 = np.argsort(-fea_ )[:300]
###第一列是标签 所以要加1
imp_fea_300 = [i + 1 for i in imp_fea_300]
imp_fea_300.insert(0,0)
data_fea_select = data.iloc[0:,imp_fea_300]