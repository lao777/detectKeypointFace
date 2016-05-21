import os
import other

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

FTRAIN = '~/detectKeypoint/dataTrain/train.csv'
FTEST = '~/detectKeypoint/dataTrain/test.csv'

RES  = '~/detectKeypoint/dataTrain/original/IdLookupTable.csv'

ORIGINALTRAIN = '~/detectKeypoint/dataTrain/original/training.csv'
ORIGINALTEST = '~/detectKeypoint/dataTrain/original/test.csv'

feaches = {
"left_eye_center_x" : 0,
"left_eye_center_y" : 1,
"right_eye_center_x" : 2,
"right_eye_center_y" : 3,
"left_eye_inner_corner_x" : 4,
"left_eye_inner_corner_y" : 5,
"left_eye_outer_corner_x" : 6,
"left_eye_outer_corner_y" : 7,
"right_eye_inner_corner_x" : 8,
"right_eye_inner_corner_y" : 9,
"right_eye_outer_corner_x" : 10,
"right_eye_outer_corner_y" : 11,
"left_eyebrow_inner_end_x" : 12,
"left_eyebrow_inner_end_y" : 13,
"left_eyebrow_outer_end_x" : 14,
"left_eyebrow_outer_end_y" : 15,
"right_eyebrow_inner_end_x" : 16,
"right_eyebrow_inner_end_y" : 17,
"right_eyebrow_outer_end_x" : 18,
"right_eyebrow_outer_end_y" : 19,
"nose_tip_x" : 20,
"nose_tip_y" : 21,
"mouth_left_corner_x" : 22,
"mouth_left_corner_y" : 23,
"mouth_right_corner_x" : 24,
"mouth_right_corner_y" : 25,
"mouth_center_top_lip_x" : 26,
"mouth_center_top_lip_y" : 27,
"mouth_center_bottom_lip_x" : 28,
"mouth_center_bottom_lip_y" : 29
}


def load(test=False, cols=None, testReal = False):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = ORIGINALTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    #print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    y = df[df.columns[:-1]].values
    y = (y - 48) / 48  # scale target coordinates to [-1, 1]
    if(test == False):
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
    y = y.astype(np.float32)

    return X, y

def load2d(test=False, cols=None, testReal2d = False):
    X, y = load(test=test, testReal = testReal2d)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def TestResault(net, is2d = False):
    if(is2d == True):
        xTest, yTest = load2d(test = True)
    else:
        xTest, yTest = load(test = True)

    yPred = net.predict(xTest)

    yPred = yPred.reshape(1, len(yPred[0])* len(yPred) )
    yTest = yTest.reshape(1, len(yTest[0])* len(yTest) )

    yPred = yPred.tolist()
    yTest = yTest.tolist()


    rms = sqrt(mean_squared_error(yTest, yPred))
    return rms * 48

def originalSize(x):
    for i in range(0, len(x)):
        for j in range(0, len(x[i])):
            x[i][j] = x[i][j] * 48 + 48
            if(x[i][j] > 96):
                x[i][j] = 96
            if(x[i][j] < 0):
                x[i][j] = 0
    return x

def WritePredict():
    X, y = load2d()
    net6.fit(X, y)

    print TestResault(net6, is2d = True)

    Xtr, yTr = load2d(testReal2d = True)

    pred = net2.predict(Xtr)
    originalSize(pred)

    res = read_csv(os.path.expanduser(RES))

    for i in range(0, 27124):
        indexFeaches = feaches[res.at[i, 'FeatureName']]
        indexSample = res.at[i, 'ImageId'] - 1
        res.at[i, 'Location'] = pred[indexSample][indexFeaches]
        
        
    

    res.to_csv("~/shad-env/detectKeypoint/resault.csv", cols=['RowId', 'Location'])

flip_indices = [
    (0, 2), (1, 3),
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (22, 24), (23, 25),
    ]

# Let's see if we got it right:
df = read_csv(os.path.expanduser(FTRAIN))
for i, j in flip_indices:
    print("# {} -> {}".format(df.columns[i], df.columns[j]))

from nolearn.lasagne import BatchIterator

class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

def float32(k):
    return np.cast['float32'](k)

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

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    hidden4_num_units=500,
    dropout4_p=0.5,  # !
    hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=100,
    verbose=1,
    eval_size=0.1
    )
net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=1000,
    verbose=1,
    )

X, y = load2d()  # load 2-d data
net2.fit(X, y)

'''


X, y = load2d()


net.fit(X, y)

other.SaveNet(net)

print TestResault(net, is2d = True)
'''


'''
net = other.LoadNet()


print TestResault(net, is2d = True)

Xtr, yTr = load2d()

pred = net.predict(Xtr)
originalSize(pred)

res = read_csv(os.path.expanduser(RES))

for i in range(0, 27124):
    indexFeaches = feaches[res.at[i, 'FeatureName']]
    indexSample = res.at[i, 'ImageId'] - 1
    res.at[i, 'Location'] = pred[indexSample][indexFeaches]

res.to_csv("~/shad-env/detectKeypoint/resault.csv", columns=['RowId', 'Location'], index=False)
'''