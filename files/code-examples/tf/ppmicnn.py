import numpy as np
import scipy.io as sio

import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Activation, concatenate, Input
from keras.layers import Conv3D, Conv2D, MaxPooling3D, MaxPooling2D, Convolution3D
# For salinecy tests
import vis
from vis.visualization import visualize_saliency, visualize_activation, visualize_cam
from vis.utils import utils
from keras import activations

from keras.layers.core import *
from keras.models import Model

def convert3Dto2D(imageData3D, method):
    # convert 3D images to 2D images
    imageSize = imageData3D.shape
    numberOfImages = len(imageData3D)
    imageData2D = np.zeros([numberOfImages,imageSize[1],imageSize[2],1])
    for nIm in range(numberOfImages):
        img = imageData3D[nIm,:,:,:,:]
        if method=='MIP':
            imgNew = np.max(img, axis=2)
            imageData2D[nIm,:,:,:] = imgNew
        else:
            continue
    return imageData2D

def loadImages(folder,nameList,varName,dims):
    # load tac data from a matlab file
    numImages = len(nameList)
    all_data = np.zeros((numImages,dims[0],dims[1],dims[2],1))
    for i in range(0,numImages):
        filename = str(nameList[i]) + '.mat'
        FA_org = sio.loadmat(folder+filename)
        FA_data = FA_org[varName]
        img_data = FA_data.copy()
        img_data = img_data.astype('float32')
        #img_data = np.reshape(img_data,(41*41,))
        all_data[i,:,:,:,0] = img_data
    return all_data

def loadDataFromMat(filename,varList):
    FA_org = sio.loadmat(filename)
    dataDict = {}
    for var in varList:
        FA_data = FA_org[var]
        tmp = FA_data.copy()
        #tmp = tmp.astype('float32')
        dataDict[var] = tmp
    return dataDict

def readDataFromCsv(filename, readNames):
    """Read data from .csv file into an array.
    Output: a numpy array"""
    import numpy as np
    data = np.genfromtxt(filename, delimiter = ',', names=readNames)
    return data

def splitDataPart(data,partVec):
    """Split input table into two tables with rows defined according to partVec.
    Return a dictionary with split train and test data"""
    sel1 = partVec==1 # train
    sel2 = partVec==2 # test
    dataPart1 = data[sel1]
    dataPart2 = data[sel2]
    dataPart = {'dataClass1':dataPart1, 'dataClass2':dataPart2}
    return dataPart

def initModelReg(imageSize):
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='elu', input_shape=imageSize+[1]))
    model.add(Conv3D(32, (3, 3, 3), activation='elu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3), activation='elu'))
    model.add(Conv3D(64, (3, 3, 3), activation='elu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='elu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

def initModelReg2D(imageSize):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=imageSize+[1]))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

def initModelCat(imageSize):
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=imageSize+[1])) #imageSize+[1]
    model.add(Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='relu'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def initModelCat2D(imageSize):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='elu', input_shape=imageSize+[1])) #imageSize+[1]
    model.add(Conv2D(32, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='elu'))
    model.add(Dense(2, activation='softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def getModelSaliency(model,input,classIdx):
    # Swap softmax with linear
    layer_idx = -1
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    grads = visualize_cam(model, layer_idx, filter_indices=classIdx, seed_input=input, grad_modifier=None) # backprop_modifier='guided' 'relu'
    return grads

def getModelActivation(model,classIdx):
    layer_idx = -1
    act = visualize_activation(model, layer_idx, filter_indices=classIdx, seed_input=None, input_range=(0, 255), backprop_modifier='guided')
    return act

def initModelRegAux(imageSize,numberOfAuxInputs):
    # initialize regression model with auxiliary inputs
    input_image = Input(shape=imageSize+[1])
    input_aux = Input(shape=(numberOfAuxInputs,))    

    x = Convolution3D(filters=32, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu')(input_image)
    x = Convolution3D(filters=32, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Convolution3D(filters=64, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu')(x)
    x = Convolution3D(filters=64, kernel_size=(3,3,3), strides=(1, 1, 1), activation='relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    conv_flat = Flatten()(x)
    conv_flat2 = Dense(64, activation='relu')(conv_flat)

    merged = concatenate([conv_flat2, input_aux])
    predictions = Dense(1, activation ='linear')(merged)

    model = Model(inputs=[input_image, input_aux], outputs=predictions)
    model.compile(loss='mean_absolute_error',optimizer='adam')
    return model


    # x = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(input_image)
    # x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    # x = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(x)
    # x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    # conv_flat = Flatten()(x)
    # merged = concatenate([conv_flat, input_aux])
    # predictions = Dense(1, activation ='sigmoid')(merged)

    # model = Model(inputs=[input_image, input_aux], outputs=predictions)
    # model.compile(loss='binary_crossentropy',optimizer='adam')
    # return model

def augmentData(dataXY, numAug):
    # determine the number of augmentations required for each class
    totalData = len(dataXY)
    dataXYaug = np.zeros((totalData*numAug,),dtype = dataXY.dtype)
    # generate class one augmentations
    addCount = 0
    for r in range(dataXY.size):
            newRow = dataXY[r]
            for nr in range(numAug):
                # replicate the row but change the image number
                # add row to the new table
                dataXYaug[addCount] = newRow 
                st = str(dataXYaug[addCount]['IMAGEID_BL']) + '-' + str(nr+1)
                dataXYaug[addCount]['IMAGEID_BL'] = np.str_(st)
                addCount = addCount + 1
    # crop the new tables to the specified size
    return dataXYaug