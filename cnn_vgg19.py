from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import scipy.misc
import os,sys,time,random,glob,pickle
import theano
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, ExpressionLayer, DenseLayer, InputLayer, DropoutLayer
from lasagne.layers import batch_norm
from lasagne.nonlinearities import rectify, softmax, sigmoid, linear, LeakyRectify
from lasagne.utils import floatX
import lasagne



def build_vggnet(net_input):
    imagenet_mean = floatX(np.array([104, 117, 123]).reshape(1, -1, 1, 1))
    net = {}
    net['input'] = InputLayer((None,3,224,224), input_var=net_input)
    net['preprocess'] = ExpressionLayer(net['input'], lambda x: x[:,::-1,:,:] - imagenet_mean, 'auto')
    net['conv1_1'] = Conv2DLayer(net['preprocess'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = Conv2DLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)
    net['conv2_1'] = Conv2DLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = Conv2DLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)
    net['conv3_1'] = Conv2DLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = Conv2DLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = Conv2DLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = Conv2DLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = MaxPool2DLayer(net['conv3_4'], 2)
    net['conv4_1'] = Conv2DLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = Conv2DLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = Conv2DLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = Conv2DLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = MaxPool2DLayer(net['conv4_4'], 2)
    net['conv5_1'] = Conv2DLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = Conv2DLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = Conv2DLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = Conv2DLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = MaxPool2DLayer(net['conv5_4'], 2)
    net['fc1'] = DenseLayer(net['pool5'],
                            num_units=100,
                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    net['dropout1'] = DropoutLayer(net['fc1'])

    net['fc2'] = DenseLayer(net['dropout1'],
                            num_units=10,
                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    net['dropout2'] = DropoutLayer(net['fc2'])
    net['output'] = DenseLayer(net['dropout2'],
                               num_units=1,
                               nonlinearity=sigmoid)
    
    params = lasagne.layers.get_all_params(net['pool5'])
    values = pickle.load(open('vgg19_normalized.pkl'))['param values']
    for p, v in zip(params, values):
        p.set_value(v)
    return net

def new_cnn(net_input):
    net={}
    net['input'] = InputLayer((None,3,100,100), input_var=net_input)
    net['conv1_1'] = Conv2DLayer(net['input'], 16, 3, pad=1, flip_filters=False)
    net['conv1_2'] = Conv2DLayer(net['conv1_1'], 16, 3, pad=1, flip_filters=False)
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)
    net['conv2_1'] = Conv2DLayer(net['pool1'], 32, 3, pad=1, flip_filters=False)
    net['conv2_2'] = Conv2DLayer(net['conv2_1'], 32, 3, pad=1, flip_filters=False)
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)
    net['conv3_1'] = Conv2DLayer(net['pool2'], 64, 3, pad=1, flip_filters=False)
    net['conv3_2'] = Conv2DLayer(net['conv3_1'], 64, 3, pad=1, flip_filters=False)
    net['pool3'] = MaxPool2DLayer(net['conv3_2'], 2)
    net['fc1'] = DropoutLayer(DenseLayer(net['pool3'],
                                         num_units=10,
                                         nonlinearity=lasagne.nonlinearities.leaky_rectify))
    net['output'] = DenseLayer(net['fc1'],
                               num_units=1,
                               nonlinearity=sigmoid)
    return net



def freeze_layers(net, layer):
    layers = lasagne.layers.get_all_layers(net[layer])
    for layer in layers:
        for param in layer.params:
            layer.params[param].discard('trainable')
    return net



