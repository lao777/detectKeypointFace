import sys
sys.path.append("/home/lao/release/lib/") 
import os
import cv2

from PIL import Image, ImageFilter
import glob


import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import lasagne
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from lasagne import layers
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet
import cPickle as pickle
import theano
from IPython.display import display

from nolearn.lasagne import BatchIterator
from PIL import ImageDraw, ImageFont
import time


FTRAIN = '/home/lao/detectKeypoint/dataHelenScale/train/'
FTEST = '/home/lao/detectKeypoint/dataHelenScale/test/'


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
def float32(k):
    return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()
def LoadNet():
    with open('net1.pickle', 'rb') as f:
        return pickle.load(f)


pointInSample = [0, 8, 16, 36, 45, 30, 48, 54]
net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 100, 100),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,  # !
    hidden4_num_units=500,
    dropout4_p=0.5,  # !
    hidden5_num_units=500,

    output_num_units=len(pointInSample) * 2,
    output_nonlinearity=None,

    update=nesterov_momentum,
    
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=100),
        ],
    
    batch_iterator_train=BatchIterator(batch_size=10),
    regression=True,
    max_epochs=800,
    verbose=1,
    eval_size=0.1
    )

net1 = LoadNet()

def Predict(image):
    imResize = cv2.resize(image,(width, height))
    imResize = np.divide(imResize, 255.) 
    for i in range(0, 3):
        for j in range(0, width):
            for k in range(0, height):
                imNew[0, i, j, k] = imResize[j, k, i]

    pred = net1.predict(imNew)
    pred = np.subtract(pred, -1)
    pred = np.multiply(pred, 0.5)
    pred = pred.reshape(len(pred[0]) / 2, 2)

    return pred


cap = cv2.VideoCapture(0)

cascPath = "/home/lao/detectKeypoint/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Capture frame-by-frame

from multiprocessing import pool
from multiprocessing.dummy import Pool as TreadPool


    

# Our operations on the frame come here
width = 100
height = 100
originalWidth = 640
originalHeight = 480
imNew = np.zeros((1, 3, width, height))
imNew = imNew.astype(np.float16)
while(True):
    ret, frame = cap.read()

    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ####rect

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )


    # Draw a rectangle around the faces


    for (x, y, w, h) in faces:
        x1 = int(x - 0.2 * w) 
        x1 = 0 if x1 < 0 else x1

        x2 = int(x + 1.2 * w)
        x2 = originalWidth if x2 > originalWidth - 1 else x2

        y1 = int(y)
        y1 = 0 if y1 < 0 else y1


        y2 = int(y + 1.3 * h)
        y2 = originalHeight if y2 > originalHeight - 1 else y2

        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        imRect = im[y1 : y2, x1 : x2]


        pred = Predict(imRect)
        

        for i in range(0, len(pointInSample)):
            pass

            cv2.circle(frame, (int(pred[i][0] * (x2 - x1) + x1 ), int(pred[i][1] * (y2 - y1) + y1 )),
            5, (0,0,255), -1)

    #end rect
    


    frame = np.fliplr(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()