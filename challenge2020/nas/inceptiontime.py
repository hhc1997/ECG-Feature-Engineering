import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)
rootPath = curPath
for i in range(1):
    rootPath = os.path.split(rootPath)[0]
print(rootPath)
sys.path.append(rootPath)
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow.keras as keras
from utils.data import loaddata
import autokeras as ak
from nas.utils.logger import Logger
from nas.utils.tools import AdvancedLearnignRateScheduler
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
(X_train, y_train), (X_val, y_val), (X_test, y_test)= loaddata()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train[:3])
Nclass = 9
classes = ['Nor', 'AF', 'I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']

# Initialize the classifier.
input_node = ak.Input()
output_node = ak.InceptionTimeBlock(type='v2')(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=30,
                   name='2018_inceptiontime_Greedy',directory='nas_result')
#
sys.stdout = Logger('nas_result/2018_inceptiontime_Greedy/log', sys.stdout)
sys.stderr = Logger('nas_result/2018_inceptiontime_Greedy/log_file', sys.stderr)		# redirect std err, if necessary

clf.tuner.search_space_summary()
# Search for the best model.

clf.fit(X_train,y_train, epochs=100, validation_data=(X_val, y_val), batch_size=16, callbacks=[keras.callbacks.EarlyStopping(patience=10),
    AdvancedLearnignRateScheduler(monitor='val_loss', patience=6, verbose=1, mode='auto', decayRatio=0.1, warmup_batches=5, init_lr=0.001)],verbose=0)

clf.tuner.results_summary()

# Evaluate the best model on the testing data.
# loss, accuracy, precision, recall, f1 = clf.evaluate(final_testset, final_testtarget)
# print('*************************----best_model----*************************')
# print('loss:', loss)
# print('accuracy:', accuracy)
# print('precision:', precision)
# print('recall:', recall)
# print('f1:', f1)

# Evaluate the best 10 models( only a convenience shortcut, recommended to retrain the models)
best_models = clf.tuner.get_best_models(num_models=10)

for i in range(10):
    # loss, accuracy, precision, recall, f1 = best_models[i][2].evaluate(final_testset, final_testtarget)
    print('*************************----best_model_'+str(i)+'----*************************')
    model = best_models[i][2]
    cvconfusion = np.zeros((Nclass, Nclass, 2))
    counter = 0
    ypred = model.predict(X_val)
    ypred = np.argmax(ypred, axis=1)
    ytrue = np.argmax(y_val, axis=1)
    cvconfusion[:, :, counter] = confusion_matrix(ytrue, ypred)
    F1 = np.zeros((9, 1))
    for i in range(9):
        F1[i] = 2 * cvconfusion[i, i, counter] / (
                np.sum(cvconfusion[i, :, counter]) + np.sum(cvconfusion[:, i, counter]))
        print("validation F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
    print("validation Overall F1 measure: {:1.4f}".format(np.mean(F1[0:9])))

    counter += 1
    ypred = model.predict(X_test)
    ypred = np.argmax(ypred, axis=1)
    ytrue = np.argmax(y_test, axis=1)
    cvconfusion[:, :, counter] = confusion_matrix(ytrue, ypred)
    F1 = np.zeros((9, 1))
    Precision = np.zeros((9, 1))
    Recall = np.zeros((9, 1))
    Accuracy = 0
    for i in range(9):
        F1[i] = 2 * cvconfusion[i, i, counter] / (
                np.sum(cvconfusion[i, :, counter]) + np.sum(cvconfusion[:, i, counter]))
        print("test F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
        Precision[i] = cvconfusion[i, i, counter] / np.sum(cvconfusion[:, i, counter])
        Recall[i] = cvconfusion[i, i, counter] / np.sum(cvconfusion[i, :, counter])
        Accuracy += cvconfusion[i, i, counter] / np.sum(cvconfusion[:, :, counter])
    print("test Overall F1 measure: {:1.4f}".format(np.mean(F1[0:9])))
# model = clf.export_model()
# model.save('nas_result/2018_inceptiontime_Greedy/bestmodel.h5')