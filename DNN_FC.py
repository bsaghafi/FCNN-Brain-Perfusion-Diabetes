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
from keras.callbacks import EarlyStopping
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
    X_train=np.load('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/AADHS_3DCNN_classifier/X_train_regional.npy')
    # X_test=np.load('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/AADHS_3DCNN_classifier/X_test_CD_gfr.npy')
    y_train=np.load('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/AADHS_3DCNN_classifier/y_train_regional.npy')
    # y_test=np.load('/project/radiology/ANSIR_lab/s174380/dev/DeepLearning/AADHS_3DCNN_classifier/y_test_CD_gfr.npy')
    # return X_train, X_test, y_train, y_test
    return X_train, y_train

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

def create_model():
    model = Sequential()

    model.add(ZeroPadding3D((1, 1, 1), input_shape=(1, 121,145,121), dim_ordering='th'))
    model.add(Convolution3D(4, 3, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(4, 3, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))

    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(32, 3, 3, 3, activation='relu', dim_ordering='th'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))

    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(16, 3, 3, 3, activation='relu', dim_ordering='th'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    # BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    # model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])
    return model

nb_conv_blocks=3
#nb_filters=16
#sz_filters_b1=11
#sz_filters_b2=5
#sz_filters_b3=3
#FC_layer_size=64

def create_model5(nb_filters,sz_filters,FC_layer_size):
    model = Sequential()

    model.add(BatchNormalization(input_shape=(1, 48, 60, 52), axis=1))
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    #model.add(ZeroPadding3D((1, 1, 1), input_shape=(1, 48, 60, 52), dim_ordering='th'))
    model.add(Convolution3D(nb_filters, sz_filters, sz_filters, sz_filters, activation='relu', dim_ordering='th', name='conv_1'))
    BatchNormalization(axis=1)
    model.add(Convolution3D(nb_filters, sz_filters, sz_filters, sz_filters, activation='relu', dim_ordering='th', name='conv_2'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))
    BatchNormalization(axis=1)

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(nb_filters*2, sz_filters, sz_filters, sz_filters, activation='relu', dim_ordering='th', name='conv_3'))
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(nb_filters*2, sz_filters, sz_filters, sz_filters, activation='relu', dim_ordering='th', name='conv_4'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))
    BatchNormalization(axis=1)

    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(nb_filters*4, sz_filters, sz_filters, sz_filters, activation='relu', dim_ordering='th', name='conv_5'))
    BatchNormalization(axis=1)
    model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    model.add(Convolution3D(nb_filters*4, sz_filters, sz_filters, sz_filters, activation='relu', dim_ordering='th', name='conv_6'))
    BatchNormalization(axis=1)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(nb_filters*8, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_7'))
    # # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(nb_filters*8, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_8'))
    # # BatchNormalization(axis=1)
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))
    #
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(nb_filters*16, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_9'))
    # # BatchNormalization(axis=1)
    # model.add(ZeroPadding3D((1, 1, 1), dim_ordering='th'))
    # model.add(Convolution3D(nb_filters*16, 3, 3, 3, activation='relu', dim_ordering='th', name='conv_10'))
    # # BatchNormalization(axis=1)
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(FC_layer_size, activation='relu'))
    BatchNormalization(axis=1)
    model.add(Dropout(0.6))
    model.add(Dense(2, activation='softmax'))

    # sgd = SGD(lr=1e-3)
    # adam = Adam(lr=1e-5)
    # add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

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
    # model.add(Dense(2, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


train_data, y_train = load_data()
from sklearn.preprocessing import Imputer, MinMaxScaler
train_data=np.asarray(train_data,dtype=np.float64)
train_data=MinMaxScaler().fit_transform(train_data)
train_target = np_utils.to_categorical(y_train, 2)

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
cvscores=[]
fscore=[]
deltaloss=[]
deltaloss2=[]
deltaloss_per_epoch=[]
# starting a timer
start = timeit.default_timer()
print train_data.shape
print train_target.shape

for train_index, test_index in kfold.split(train_data, y_train):
    nepoch = 1000
    bsize = 10
    adam = Adam(lr=1e-3, beta_1=0.5)
    rms = RMSprop(lr=1e-3, rho=0.5)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.5, nesterov=True)
    model = model_DNN()
    callbacks = [EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=2, mode='auto')]
    history= model.fit(train_data[train_index], train_target[train_index],
              batch_size=bsize, nb_epoch=nepoch, verbose=1,
              validation_data=(train_data[test_index], train_target[test_index]),
              callbacks=callbacks,
              shuffle=True)
    score = model.evaluate(train_data[test_index], train_target[test_index], batch_size=bsize, verbose=0)
    cvscores.append(score[1] * 100)
    Loss = history.history['loss']
    deltaloss.append(100 * (Loss[0] - Loss[-1]) / Loss[0])
    deltaloss2.append(100 * (Loss[1] - Loss[-1]) / Loss[1])
    deltaloss_per_epoch.append(dec_per_epoch(Loss))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    print("%s: %.2f%%" % ('% loss decrease', 100 * (Loss[0] - Loss[-1]) / Loss[0]))
    print("%s: %.2f%%" % ('% loss decrease2', 100 * (Loss[1] - Loss[-1]) / Loss[1]))
    print("%s: %.2f%%" % ('% delta loss per epoch', dec_per_epoch(Loss)))
    # plot_training_loss_acc(history)
    break

#
print '**************Overall Results:**************'
print("%s: %.2f%% (+/- %.2f%%)" % ('Average accuracy', np.mean(cvscores), np.std(cvscores)))
MDL = np.mean(deltaloss)
print("%s: %.2f%%" % ('Average loss decrease', MDL))
print("%s: %.2f%%" % ('Average loss decrease2', np.mean(deltaloss2)))
MDLPP = np.mean(deltaloss_per_epoch)
print("%s: %.2f%%" % ('Average DLoss per epoch', MDLPP))
# print np.mean(deltaloss)

stop = timeit.default_timer()
print 'Total run time in mins: {}'.format((stop - start) / 60)


