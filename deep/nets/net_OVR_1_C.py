from lasagne import layers, nonlinearities
from nolearn.lasagne import NeuralNet
import theano

import util
from iterators import ParallelBatchIterator, AugmentingParallelBatchIterator
from learning_rate import AdjustVariable
from imageio import ImageIO
import stats
from modelsaver import ModelSaver
from params import *
import numpy as np
from lasagne.nonlinearities import LeakyRectify
from labels import OvRConverter

def define_net_specific_parameters():
    params.START_LEARNING_RATE = 0.005
    params.BATCH_SIZE = 64
    params.MULTIPROCESS = False
    params.REGRESSION = False

def define_net():
    define_net_specific_parameters()

    io = ImageIO()

    # Read pandas csv labels
    y = util.load_labels()

    if params.SUBSET is not 0:
        y = y[:params.SUBSET]

    X = np.arange(y.shape[0])

    mean, std = io.load_mean_std(circularized=params.CIRCULARIZED_MEAN_STD)
    keys = y.index.values

    # OVR
    ovr = OvRConverter(number = 1)
    y = ovr.transform(y)

    if params.AUGMENT:
        train_iterator = AugmentingParallelBatchIterator(keys, params.BATCH_SIZE, std, mean, y_all = y)
    else:
        train_iterator = ParallelBatchIterator(keys, params.BATCH_SIZE, std, mean, y_all = y)

    test_iterator = ParallelBatchIterator(keys, params.BATCH_SIZE, std, mean, y_all = y)

    if params.REGRESSION:
        y = util.float32(y)
        y = y[:, np.newaxis]

    if 'gpu' in theano.config.device:
        # Half of coma does not support cuDNN, check whether we can use it on this node
        # If not, use cuda_convnet bindings
        from theano.sandbox.cuda.dnn import dnn_available
        if dnn_available() and not params.DISABLE_CUDNN:
            from lasagne.layers import dnn
            Conv2DLayer = dnn.Conv2DDNNLayer
            MaxPool2DLayer = dnn.MaxPool2DDNNLayer
        else:
            from lasagne.layers import cuda_convnet
            Conv2DLayer = cuda_convnet.Conv2DCCLayer
            MaxPool2DLayer = cuda_convnet.MaxPool2DCCLayer
    else:
        Conv2DLayer = layers.Conv2DLayer
        MaxPool2DLayer = layers.MaxPool2DLayer

    Maxout = layers.pool.FeaturePoolLayer

    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            ('conv2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            ('conv3', Conv2DLayer),
            ('pool3', MaxPool2DLayer),
            ('conv4', Conv2DLayer),
            ('pool4', MaxPool2DLayer),
            ('dropouthidden1', layers.DropoutLayer),
            ('hidden1', layers.DenseLayer),
            ('maxout1', Maxout),
            ('dropouthidden2', layers.DropoutLayer),
            ('hidden2', layers.DenseLayer),
            ('maxout2', Maxout),
            ('dropouthidden3', layers.DropoutLayer),
            ('output', layers.DenseLayer),
        ],

        input_shape=(None, params.CHANNELS, params.PIXELS, params.PIXELS),

        conv1_num_filters=32, conv1_filter_size=(8, 8), conv1_pad = 1, conv1_stride=(2, 2), pool1_pool_size=(2, 2), pool1_stride=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2), pool2_stride=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2), pool3_stride=(2, 2),
        conv4_num_filters=256, conv4_filter_size=(3, 3), pool4_pool_size=(2, 2), pool4_stride=(2, 2),

        hidden1_num_units=1024,
        hidden2_num_units=1024,

        dropouthidden1_p=0.5,
        dropouthidden2_p=0.5,
        dropouthidden3_p=0.5,

        maxout1_pool_size=2,
        maxout2_pool_size=2,

        output_num_units=1 if params.REGRESSION else 2,
        output_nonlinearity=None if params.REGRESSION else nonlinearities.softmax,

        conv1_nonlinearity = LeakyRectify(0.1),
        conv2_nonlinearity = LeakyRectify(0.1),
        conv3_nonlinearity = LeakyRectify(0.1),
        conv4_nonlinearity = LeakyRectify(0.1),
        hidden1_nonlinearity = LeakyRectify(0.1),
        hidden2_nonlinearity = LeakyRectify(0.1),

        update_learning_rate=theano.shared(util.float32(params.START_LEARNING_RATE)),
        update_momentum=theano.shared(util.float32(params.MOMENTUM)),
        custom_score=('kappa', util.quadratic_kappa),

        regression=params.REGRESSION,
        batch_iterator_train=train_iterator,
        batch_iterator_test=test_iterator,
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=params.START_LEARNING_RATE),
            stats.Stat(),
            ModelSaver()
        ],
        max_epochs=400,
        verbose=1,

        # Only relevant when create_validation_split = True
        eval_size=0.1,

        # Need to specify splits manually like indicated below!
        create_validation_split=params.SUBSET>0,
    )

    # It is recommended to use the same training/validation split every model for ensembling and threshold optimization
    #
    # To set specific training/validation split:
    net.X_train = np.load(params.IMAGE_SOURCE + "/X_train.npy")
    net.X_valid = np.load(params.IMAGE_SOURCE + "/X_valid.npy")
    net.y_train = np.load(params.IMAGE_SOURCE + "/y_train.npy")
    net.y_valid = np.load(params.IMAGE_SOURCE + "/y_valid.npy")

    return net, X, y
