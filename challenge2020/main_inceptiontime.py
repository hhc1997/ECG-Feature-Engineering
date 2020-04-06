from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.io
import os, time
from sklearn.metrics import confusion_matrix
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.loss import my_loss, f1
from utils.data import loaddata
from utils.tools import AdvancedLearnignRateScheduler
from models.inceptiontime import Classifier_INCEPTION
from keras.callbacks import TensorBoard
from split_data import get_train_val_test
def model_eval(X_train, y_train, X_val, y_val, X_test, y_test):
    batch = 16
    epochs = 100

    # classes = ['1', '2', '3','4','5','6','7','8','9']
    classes = ['Nor', 'AF', 'I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
    # 1: 918   Normal                                        0  1
    # 2: 1098  Atrial fibrillation (AF)                      1  1
    # 3: 704   First-degree atrioventricular block (I-AVB)   2  1.2
    # 4: 207   Left bundle branch block (LBBB)               3  5
    # 5: 1695  Right bundle branch block (RBBB)              4  0.7
    # 6: 574   Premature atrial contraction (PAC)            5  2
    # 7: 653   Premature ventricular contraction (PVC)       6  1.5
    # 8: 826   ST-segment depression (STD)                   7  1.2
    # 9: 202   ST-segment elevated (STE)                     8  5
    Nclass = len(classes)
    cvconfusion = np.zeros((Nclass, Nclass, 2))
    cvscores = []
    counter = 0

    # Load model
    model = Classifier_INCEPTION(bottleneck_size=32, depth=6, kernel_size=40, nb_filters=32, type='inceptiontime_v2',head_dropout_rate=0.5).model
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam,
                  loss= 'categorical_crossentropy',
                  metrics=['accuracy',f1])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        AdvancedLearnignRateScheduler(monitor='val_loss', patience=6, verbose=1, mode='auto',
                                      decayRatio=0.1),
        ModelCheckpoint('./result/result-inceptiontime/best_model.hdf5',
                        monitor='val_loss', save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger('./result/result-inceptiontime/log.csv', separator=',',
                                  append=True),
        TensorBoard(log_dir='./result/result-inceptiontime/tensorboard-log', write_graph=True, histogram_freq=1)
    ]

    train_number = int(len(X_train) / 16) * 16
    val_number = int(len(X_val) / 16) * 16
    model.fit(X_train[0:train_number], y_train[0:train_number],
              validation_data=(X_val[0:val_number], y_val[0:val_number]),
              epochs=epochs, batch_size=batch, callbacks=callbacks)

    # Evaluate best trained model
    model.load_weights('./result/result-inceptiontime/best_model.hdf5')
    ypred = model.predict(X_val)
    ypred = ypred
    ypred = np.argmax(ypred, axis=1)
    ytrue = np.argmax(y_val, axis=1)
    cvconfusion[:, :, counter] = confusion_matrix(ytrue, ypred)
    F1 = np.zeros((9, 1))
    for i in range(9):
        F1[i] = 2 * cvconfusion[i, i, counter] / (
                np.sum(cvconfusion[i, :, counter]) + np.sum(cvconfusion[:, i, counter]))
        print("validation F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
    cvscores.append(np.mean(F1[0:9]) * 100)
    print("validation Overall F1 measure: {:1.4f}".format(np.mean(F1[0:9])))

    counter += 1
    inference_start_time = time.time()
    ypred = model.predict(X_test)
    inference_end_time = time.time()
    print('time for inference: ', inference_end_time - inference_start_time)
    ypred = ypred
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
    import csv
    csvFile = open('./result/result-inceptiontime/result.csv', 'a', newline='')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow([str(np.mean(F1[0:9])), str(np.mean(Precision[0:9])), str(np.mean(Recall[0:9])), Accuracy])
    csvFile.close()
    cvscores.append(np.mean(F1[0:9]) * 100)
    print("test Overall F1 measure: {:1.4f}".format(np.mean(F1[0:9])))
    print(cvconfusion)
    scipy.io.savemat('./result/result-inceptiontime/cvconfusion.mat', mdict={'cvconfusion': cvconfusion.tolist()})
    return model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed = 7
    np.random.seed(seed)
    tvt = get_train_val_test()
    (X_train, y_train), (X_val, y_val), (X_test, y_test)= loaddata(tvt[0],tvt[1],tvt[2])
    print('!!!!!!!!!!')
    print(X_test.shape)

    start_time = time.time()

    model = model_eval(X_train, np.asarray(y_train), X_val, np.asarray(y_val), X_test, np.asarray(y_test))

    end_time = time.time()

    print('time for train: ', end_time - start_time)

    # Outputing results of cross validation
    matfile = scipy.io.loadmat('./result/result-inceptiontime/cvconfusion.mat')
    cv = matfile['cvconfusion']
    F1mean = np.zeros(cv.shape[2])
    outputname = ['validation', 'test']
    for j in range(cv.shape[2]):
        classes = ['Nor', 'AF', 'I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
        F1 = np.zeros((9, 1))
        for i in range(9):
            F1[i] = 2 * cv[i, i, j] / (np.sum(cv[i, :, j]) + np.sum(cv[:, i, j]))
            print(outputname[j], " - F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
        F1mean[j] = np.mean(F1[0:9])
        print("mean F1 measure for: {:1.4f}".format(F1mean[j]))
    print("Overall F1 : {:1.4f}".format(np.mean(F1mean)))


