from lasagne import layers, nonlinearities
from nolearn import NeuralNet
import theano
import theano.tensor as T
from params import *
from util import *
from iterators import ScalingBatchIterator
from learning_rate import AdjustVariable
from early_stopping import EarlyStopping
from imageio import ImageIO

# Import cuDNN if using GPU
if USE_GPU:
	from lasagne.layers import dnn
	Conv2DLayer = dnn.Conv2DDNNLayer
	MaxPool2DLayer = dnn.MaxPool2DDNNLayer
else:
	Conv2DLayer = layers.Conv2DLayer
	MaxPool2DLayer = layers.MaxPool2DLayer

# Fix seed
np.random.seed(42)

def fit():
	# Load complete data set and mean into memory
	# If you don't have enough memory to do this, lower the amount of samples that are being used in imageio.py
	# This will be changed to work with disk streaming later
	io = ImageIO()
	X, y = io.load_train_full()
	mean, std = io.load_mean_std()

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
			('hidden2', layers.DenseLayer),
			('output', layers.DenseLayer),
			],

		input_shape=(None, CHANNELS, PIXELS, PIXELS),

		conv1_num_filters=32, conv1_filter_size=(8, 8), conv1_pad = 1, conv1_stride = (2, 2), pool1_pool_size=(2, 2), pool1_stride = (2, 2),
		conv2_num_filters=64, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2), pool2_stride = (2, 2),
		conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2), pool3_stride = (2, 2),
		conv4_num_filters=256, conv4_filter_size=(3, 3), pool4_pool_size=(2, 2), pool4_stride = (2, 2),

		hidden1_num_units=512,
		dropouthidden1_p=0.5,
		hidden2_num_units=512,

		output_num_units=5,
		output_nonlinearity=nonlinearities.softmax,

		update_learning_rate=theano.shared(float32(START_LEARNING_RATE)),
		update_momentum=theano.shared(float32(MOMENTUM)),

		regression=False,
		batch_iterator_train=ScalingBatchIterator(batch_size=BATCH_SIZE, mean=mean, std=std),
		batch_iterator_test=ScalingBatchIterator(batch_size=BATCH_SIZE, mean=mean, std=std),
		on_epoch_finished=[
			#AdjustVariable('update_learning_rate', start=START_LEARNING_RATE),
			#EarlyStopping(patience=20),
		],
		max_epochs=500,
		verbose=1,
		eval_size=0.2,
	)

	net.fit(X, y)

if __name__ == "__main__":
	fit()
