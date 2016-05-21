import os

from PIL import Image, ImageFilter
import sys, glob

import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import lasagne
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import cPickle as pickle
import theano
from IPython.display import display

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

FTRAIN = '/home/lao/detectKeypoint/dataHelenScale/train/'
FTEST = '/home/lao/detectKeypoint/dataHelenScale/test/'

countSampleInTrain = 1000


X = []
i = 0

for infile in glob.glob(FTRAIN + "*.jpg"):
    im = Image.open(infile)
    i = i + 1
    r,g,b = im.split()

    r = list(r.getdata())
    g = list(g.getdata())
    b = list(b.getdata())
    tmp = np.array((r, g, b))
    tmp = tmp.reshape(3, 100, 100)
    X.append(tmp.tolist())
    

    if i == countSampleInTrain:
        break

X = np.array(X)
X = np.divide(X, 255.) 
X = X.astype(np.float16)

df = read_csv(FTRAIN +"res.csv", header = None)
y = df.as_matrix()
lenY = len(y)
for i in range(countSampleInTrain, lenY):
    y = np.delete(y, countSampleInTrain, 0)
    
y = np.divide(y, 50.) 
y = np.subtract(y, 1)
y = y.astype(np.float16)

yN = []
for i in range(0, len(y)):
    yN.append([y[i][60], y[i][61]])
    
yN = np.array(yN)


net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        #('conv2', layers.Conv2DLayer),
        #('pool2', layers.MaxPool2DLayer),
        
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 100, 100),
    
    conv1_num_filters=100, conv1_filter_size=(15, 15), conv1_stride=(5, 5),
    pool1_pool_size=(9, 9),
    #conv2_num_filters=64, conv2_filter_size=(7, 7), conv2_stride=(3, 3),
    #pool2_pool_size=(2, 2),
    
    hidden5_num_units=50,
    output_num_units=136, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=40,
    verbose=1,
    )

net1.fit(X, y)