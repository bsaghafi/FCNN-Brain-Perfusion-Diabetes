# Purpose: Train and test a fully connected DNN model to predict the level of regional brain perfusion from diabetes measures.
# Inputs: train and test data in the form of numpy arrays
# Outputs: dichotomized delta FA maps and their class labels
# Date: 07/14/2017
# Author: Behrouz Saghafi
import numpy as np

np.random.seed(2016)

import os
import glob
import datetime
import time
import timeit
import warnings
import theano
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, KFold
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras import __version__ as keras_version
import scipy.io as sio
from sklearn.metrics import log_loss, accuracy_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
## Loading Train Data

def get_binary_metrics(expected_labels, our_labels):

   # sensitivity
   recall = recall_score(expected_labels, our_labels)
   print("%s: %.2f%%" % ('Sensitivity', recall * 100))
   # print '=========================='

   # specificity
   cm = confusion_matrix(expected_labels, our_labels)
   tn, fp, fn, tp = cm.ravel()
   specificity = tn / float(tn + fp)
   print("%s: %.2f%%" % ('Specificity', specificity * 100))
   print cm


   # roc_auc_score
   roc = roc_auc_score(expected_labels, our_labels)
   print("%s: %.2f%%" % ('ROC_AUC sore', roc * 100))

   # f1 score
   f1score = f1_score(expected_labels, our_labels)
   print("%s: %.2f%%" % ('F1 Score', f1score * 100))
   # print '=========================='

   accuracy = accuracy_score(expected_labels, our_labels)
   print("%s: %.2f%%" % ('Accuracy', accuracy * 100))
   print '=========================='

   return recall, specificity, roc, f1score, accuracy

# X_train, X_test, y_train, y_test = load_data()
def load_data():
    X_train=np.load('X_train_regional.npy')
    X_test=np.load('X_test_regional.npy')
    y_train=np.load('y_train_regional.npy')
    y_test=np.load('y_test_regional.npy')
    return X_train, X_test, y_train, y_test
    #return X_train, y_train

# Reshaping the Data

def read_and_normalize_data():
    X_train,X_test,y_train,y_test=load_data()
    print('Reshaping Test Data')
    print('Convert to numpy...')

    train_data = np.array(X_train,dtype=np.int16)
    y_train = np.array(y_train,dtype=np.uint8)
    print('Reshape...')
    #train_data = train_data.transpose((3,0,1,2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = (train_data-np.amin(train_data)) / (np.amax(train_data)-np.amin(train_data))
    # train_target = y_train

    train_target = np_utils.to_categorical(y_train, 2)

    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2], train_data.shape[3])

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')

    print('Reshaping Test Data')
    print('Convert to numpy...')
    #xx = X_test
    test_data = np.array(X_test, dtype=np.int16)
    test_target_pred = np.array(y_test, dtype=np.uint8)

    print('Reshape...')
    #test_data = test_data.transpose((3,0,1,2))

    print('Convert to float...')
    test_data = test_data.astype('float32')
    test_data = (test_data - np.amin(train_data)) / (np.amax(train_data) - np.amin(train_data))
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2], test_data.shape[3])

    test_target = np_utils.to_categorical(test_target_pred, 2)

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'Test samples')
    return train_data, train_target, test_data, test_target, y_train


# plotting training performance:
def plot_training_loss_acc(history):
    #with plt.style.context(('seaborn-talk')):

        fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
        fig.set_facecolor('white')
        axs[0].plot(history.history['loss'], label='Train Set', color='red')
        axs[0].plot(history.history['val_loss'], label='Validation Set', color='blue')
        axs[0].legend(loc='upper right')
        axs[0].set_ylabel('Log Loss')
        #axs[0].set_xlabel('Epochs')
        axs[0].set_title('Training Performance')

        axs[1].plot(history.history['acc'], label='Train Set', color='red')
        axs[1].plot(history.history['val_acc'], label='Validation Set', color='blue')
        axs[1].legend(loc='lower right')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_xlabel('Epochs')  # used for both subplots
        plt.show()
        fig.savefig('learning_curves_model2_cubic.png', bbox_inches='tight')
# Creating Model

def model_DNN0():
    model = Sequential()
    model.add(Dense(14, input_dim=14, activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    adam = Adam(lr=1e-3, beta_1=0.5)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_DNN():
    model = Sequential()
    model.add(Dense(16, input_dim=16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

start=timeit.default_timer()
train_data, test_data, y_train, y_test = load_data()
from sklearn.preprocessing import MinMaxScaler,StandardScaler
train_data=np.asarray(train_data,dtype=np.float64)
min_max_scaler=MinMaxScaler()
# min_max_scaler=StandardScaler()
train_data=min_max_scaler.fit_transform(train_data)
train_target = np_utils.to_categorical(y_train, 2)

test_data=np.asarray(test_data,dtype=np.float64)
test_data=min_max_scaler.transform(test_data)
test_target = np_utils.to_categorical(y_test, 2)

#k_fold=KFold(10)
#stratified=StratifiedKFold(5)
num_fold=0
sum_score=0
models=[]
acc=[]
score=[]

def dec_per_epoch(Loss):
    delta_loss=[]
    for i in range(1,len(Loss)-1):
        delta_loss.append(100 * (Loss[i] - Loss[i+1]) / Loss[i])
    return np.mean(delta_loss)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
# kfold = KFold(n_splits=92)
cvscores=[]
best_epoch_history=[]
fscore=[]
deltaloss=[]
deltaloss2=[]
deltaloss_per_epoch=[]
print train_data.shape
print train_target.shape

nepoch = 1000
bsize = 10
for train_index, test_index in kfold.split(train_data, y_train):
    adam = Adam(lr=1e-3, beta_1=0.5)
    # rms = RMSprop(lr=1e-3, rho=0.5)
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.5, nesterov=True)
    model = model_DNN()
    # callbacks = [EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=2, mode='auto')]
    best_val=0
    best_epoch=None
    best_model=None
    counter=0
    for epoch in range(1,nepoch+1):
            callbacks = [ModelCheckpoint("/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/AADHS_3DCNN_classifier/best_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=15)]
            if counter>15:
                break
            print 'Epoch=', epoch
            history=model.fit(train_data[train_index], train_target[train_index], batch_size=bsize, nb_epoch=1, verbose=2, validation_data=(train_data[test_index], train_target[test_index]), callbacks=callbacks, shuffle=True)
            val_acc=history.history['val_acc'][0]
            print 'val_acc=', val_acc
            if val_acc > best_val:
                    best_val = val_acc
                    best_epoch = epoch
                    best_model = model
                    counter=0
            else:
                counter=counter+1
            print '\n'

    # score = best_model.evaluate(train_data[test_index], train_target[test_index], verbose=0)
    cvscores.append(best_val * 100)
    best_epoch_history.append(best_epoch)
    # print("%s: %.2f%%" % (best_model.metrics_names[1], score[1] * 100))
    print best_val
    print '\n'
    # plot_training_loss_acc(history)
    # break

#
print '**************Validation Results:**************'
print cvscores
print best_epoch_history
print("%s: %.2f%% (+/- %.2f%%)" % ('Average accuracy', np.mean(cvscores), np.std(cvscores)))
print("median epoch=", np.median(best_epoch_history))
stop = timeit.default_timer()
print 'Total run time in mins: {}'.format((stop - start) / 60)

# print '**************Testing on unseen data:**************'
# adam = Adam(lr=1e-3, beta_1=0.5)
# # rms = RMSprop(lr=1e-3, rho=0.5)
# # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.5, nesterov=True)
# model = model_DNN()
# history= model.fit(train_data, train_target,
#               batch_size=bsize, nb_epoch=100, verbose=1,
#               shuffle=True)
# score = model.evaluate(test_data, test_target, batch_size=bsize, verbose=0)
# acc=score[1]
# print acc
# predicted_labels = model.predict(test_data, batch_size=bsize, verbose=0)
# our_labels=np.argmax(predicted_labels, axis=1)
# expected_labels=np.argmax(test_target, axis=1)
# get_binary_metrics(expected_labels, our_labels)


## permutation tests using Wrapper for Sklearn:
#starting a timer
# import timeit
# start = timeit.default_timer()
# expected_label_train=np.argmax(train_target, axis=1)
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.wrappers.scikit_learn import BaseWrapper
# import copy
# def custom_get_params(self, **params):
#     res = copy.deepcopy(self.sk_params)
#     res.update({'build_fn': self.build_fn})
#     return res
# BaseWrapper.get_params = custom_get_params
# # exported_model = KerasClassifier(build_fn=model_DNN, batch_size=bsize, verbose=0)
# exported_model = KerasClassifier(build_fn=model_DNN, nb_epoch=50, batch_size=bsize, verbose=1, shuffle=True)
# # # evaluate using 10-fold cross validation
# # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# # results = cross_val_score(model, X, Y, cv=kfold)
# # print(results.mean())
# from sklearn.model_selection import permutation_test_score
# score, permutation_scores, pvalue = permutation_test_score(exported_model, train_data, expected_label_train, cv=3, scoring="accuracy", n_permutations=10000, n_jobs=-1)
# print("Classification score %s (pvalue : %s)" % (score, pvalue))
# stop = timeit.default_timer()
# print 'Total run time for this fold: {}'.format(stop - start)

# # permutation tests (my code):
# import timeit
# start = timeit.default_timer()
# n_permutations=10000
# permutation_scores=[]
# # print test_target
# from sklearn.utils import shuffle
# for i in range(n_permutations):
#     train_target_randomized = shuffle(train_target, random_state=0)
#     test_target_randomized = shuffle(test_target, random_state=0)
#     # print test_target_randomized
#     adam = Adam(lr=1e-3, beta_1=0.5)
#     model = model_DNN()
#     history = model.fit(train_data, train_target_randomized,
#                         batch_size=bsize, nb_epoch=50, verbose=0,
#                         shuffle=True)
#     pscore = model.evaluate(test_data, test_target_randomized, batch_size=bsize, verbose=1)
#     permutation_scores.append(pscore[1])
#
# permutation_scores = np.array(permutation_scores)
# pvalue = (np.sum(permutation_scores >= acc) + 1.0) / (n_permutations + 1)
# np.save('permutation_scores2',permutation_scores)
# np.save('score',acc)
# print ('pvalue=',pvalue)
# stop = timeit.default_timer()
# print 'Total run time for this fold: {}'.format(stop - start)
#
#
# # the histogram of the data
# import numpy as np
# permutation_scores=np.load('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/AADHS_3DCNN_classifier/permutation_scores.npy')
# acc=np.load('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/AADHS_3DCNN_classifier/score.npy')
# import matplotlib.pyplot as plt
# result=plt.hist(permutation_scores*100, 11)
# plt.axvline(acc*100, color='r', linestyle='dashed')
# plt.xlabel('Accuracy(%)')
# plt.ylabel('Counts(#)')
# plt.title('Accuracy Histogram Plot (Permutation Analysis)')
# plt.axis([0, 100, 0, 200])
# plt.grid(True)
# plt.show()


