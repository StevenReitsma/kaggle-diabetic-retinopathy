import numpy as np
from lasagne import layers, nonlinearities
from nolearn import NeuralNet
import theano
from params import *
import util
from iterators import ScalingBatchIterator, ParallelBatchIterator
from learning_rate import AdjustVariable
from early_stopping import EarlyStopping
from imageio import ImageIO
from skll.metrics import kappa
from iterators import DataAugmentationBatchIterator
import redis

# Import cuDNN if using GPU
if USE_GPU:
    from lasagne.layers import dnn
    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
else:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

Maxout = layers.pool.FeaturePoolLayer

# Fix seed
np.random.seed(42)


def fit():
    io = ImageIO()
    # Read pandas csv labels
    y = util.load_labels()

    y = y[:1000]

    X = np.arange(y.shape[0])

    #mean, std = io.load_mean_std(r)
    mean, std = None, None
    keys = y.index.values

    train_iterator = ParallelBatchIterator(keys, y, None, BATCH_SIZE, std, mean)
    test_iterator = ParallelBatchIterator(keys, y, None, BATCH_SIZE, std, mean)

    if REGRESSION:
        y = util.float32(y)
        y = y[:, np.newaxis]

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

        input_shape=(None, CHANNELS, PIXELS, PIXELS),

        conv1_num_filters=32, conv1_filter_size=(8, 8), conv1_pad=1, conv1_stride=(2, 2), pool1_pool_size=(2, 2), pool1_stride=(2, 2),
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

        output_num_units=1 if REGRESSION else 5,
        output_nonlinearity=None if REGRESSION else nonlinearities.softmax,

        update_learning_rate=theano.shared(util.float32(START_LEARNING_RATE)),
        update_momentum=theano.shared(util.float32(MOMENTUM)),
        custom_score=('weighted kappa', lambda true, predicted: kappa(
            true, predicted, weights='quadratic')),

        regression=REGRESSION,
        batch_iterator_train=train_iterator,
        batch_iterator_test=test_iterator,
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=START_LEARNING_RATE),
            EarlyStopping(patience=50),
        ],
        max_epochs=500,
        verbose=1,
        eval_size=0.1,
    )

    net.fit(X, y)

    if REGRESSION:
    	hist, _ = np.histogram(np.minimum(4, np.maximum(0, np.round(net.predict_proba(X)))), bins=5)
    	true, _ = np.histogram(y.squeeze(), bins=5)
    	print "Distribution over class predictions on training set:", hist / float(y.shape[0])
    	print "True distribution: ",  true / float(y.shape[0])

if __name__ == "__main__":
    fit()
