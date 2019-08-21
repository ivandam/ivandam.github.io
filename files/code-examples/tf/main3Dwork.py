
#%%
import keras as kr
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv3D, MaxPooling3D
from keras.optimizers import SGD
from numpy.random import seed
from keras import backend as K

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.io as sio

import random
import time
from tensorflow import set_random_seed

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#%% define some parameters for the script
numEpochs = 50
numEpochsPerSave = 5
batchSize = 64
trainNum = 1900
testNum = 100

saveDir = '/home/ivanklyuzhin/DATA/TG/newrunBig/'
loadDir = '/home/ivanklyuzhin/DATA/TG/newrunBig/'
#%%
# read matlab images
def loadImages(folder,varname,numimages):
    # load tac data from a matlab file
    all_data = np.zeros((numimages,64,64,64,1))
    for i in range(0,numimages):
        filename = "img-" + str(i+1) + '.mat'
        FA_org = sio.loadmat(folder+filename)
        FA_data = FA_org[varname] #numpy.ndarray [32 48 16]
        img_data = FA_data.copy()
        img_data = img_data.astype('float32')
        #img_data = np.reshape(img_data,(41*41,))
        all_data[i,:,:,:,0] = img_data
    return all_data


#%%
def initModel(modelType):
    model = Sequential()
    if modelType=='simple':
        model.add(Conv3D(16, (5, 5, 5), activation='elu', input_shape=(64, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(4, 4, 4)))

        model.add(Flatten())
        model.add(Dense(16, activation='elu'))
        model.add(Dense(8, activation='elu'))
        
    if modelType=='intermediate':
        model.add(Conv3D(16, (5, 5, 5), activation='elu', input_shape=(64, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(8, (5, 5, 5), activation='elu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        
        model.add(Flatten())
        model.add(Dense(16, activation='elu'))
        model.add(Dense(8, activation='elu'))

    if modelType=='advanced':
        model.add(Conv3D(16, (5, 5, 5), activation='elu', input_shape=(64, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(8, (5, 5, 5), activation='elu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())
        model.add(Dense(16, activation='elu'))
        model.add(Dense(8, activation='elu'))

    if modelType=='pro':
        model.add(Conv3D(16, (5, 5, 5), activation='elu', input_shape=(64, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())
        model.add(Dense(16, activation='elu'))
        model.add(Dense(8, activation='elu'))

    if modelType=='mega':
        model.add(Conv3D(16, (5, 5, 5), activation='elu', input_shape=(64, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())
        model.add(Dense(16, activation='elu'))
        model.add(Dense(8, activation='elu'))

    if modelType=='ultra':
        model.add(Conv3D(16, (5, 5, 5), activation='elu', input_shape=(64, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(Conv3D(8, (3, 3, 3), activation='elu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())
        model.add(Dense(16, activation='elu'))
        model.add(Dense(8, activation='elu'))

    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

# check prediction output
def checkModelOutput(y_pred):
    # checks if all outputs are nearly the same
    diff01 = abs(y_pred[0]-y_pred[1])/abs(y_pred[0])
    diff12 = abs(y_pred[1]-y_pred[2])/abs(y_pred[1])
    diff23 = abs(y_pred[2]-y_pred[3])/abs(y_pred[2])
    if ((diff01 < 0.01) and (diff12 < 0.01) and (diff23 < 0.01)):
        return False
    else:
        return True


#%%
filename = "dataCombined.mat"
folder = loadDir
FA_org = sio.loadmat(folder+filename)
FA_data = FA_org["data"] #numpy.ndarray [32 48 16]
y_data_all = FA_data.copy()
y_data_all = y_data_all.astype('float32')
#img_data = np.reshape(img_data,(41*41,))
y_data_all = y_data_all[0:(trainNum+testNum),:]
y_data_all[:,20] = -y_data_all[:,20] # inver inf1

#%%
img = loadImages(loadDir + 'images/','img',trainNum+testNum)

#%%
print(y_data_all.shape)
print(img.shape)

#%%
x_train = img[0:trainNum,:,:,:]
x_test = img[trainNum:(trainNum+testNum),:,:,:]
print(x_train.shape)
print(x_test.shape)

modelNames = ['simple','intermediate','advanced','pro','mega','ultra']


#%%

start = time.time()

# these three lines are required for reproducibility, doesnt work with more than 1 thread
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

seed(1)
set_random_seed(1)
random.seed(1)

for fIndex in range(0,21):
    print('=========================')
    print('Running feature ' + str(fIndex+1))
    print('=========================')
    y_data = y_data_all[0:(trainNum+testNum),fIndex]
    y_train = y_data[0:trainNum]
    y_test = y_data[trainNum:(trainNum+testNum)]
    
    for mIndex in range(0,len(modelNames)):
	print(bcolors.OKBLUE + 'Running Model name:' + modelNames[mIndex] + bcolors.ENDC)
        success = False
        while not(success):
            model = initModel(modelNames[mIndex])
            for epIndex in range(0,numEpochs/numEpochsPerSave):
                hist = model.fit(x_train, y_train, batch_size = batchSize, epochs = numEpochsPerSave, shuffle=False, validation_data=[x_test,y_test])
                loss = hist.history['loss']
                loss_val = hist.history['val_loss']
                y_pred = model.predict(x_test)
                success = checkModelOutput(y_pred)
                if not(success):
		    print(bcolors.WARNING + 'Prediction failure, re-training the model' + bcolors.ENDC)
                    break
                filename = saveDir + 'feature-' + str(fIndex) + '-model-' + str(mIndex+1) + '-ep-' + str((epIndex+1)*numEpochsPerSave) + '.mat'
                sio.savemat(filename,{'y_pred':y_pred, 'y_test':y_test, 'loss':loss, 'loss_val':loss_val})

end = time.time()
print('Elapsed time:')
print(end - start)
