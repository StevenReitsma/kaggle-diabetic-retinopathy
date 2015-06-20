import numpy as np
from params import *
from subprocess import call
from iterators import TTABatchIterator
from imageio import ImageIO
import util
from math import ceil
import argparse
import importlib
from lasagne.layers import get_output, InputLayer
import theano
from nolearn.lasagne import NeuralNet

# Define so that it can be pickled
def quadratic_kappa(true, predicted):
    return kappa(true, predicted, weights='quadratic')

def weighted_round(predictions, W):
	dim = predictions.reshape((predictions.shape[0], 1))
	rep = dim.repeat(W.shape[0], axis = 1)
	delta = W - rep

	# Remove negatives
	delta[(delta < 0)] = np.inf

	preds = np.argmin(delta, axis = 1)

	return preds

def get_iter_func(model):
	last_hidden = list(model.layers_.values())[-2]
	output = list(model.layers_.values())[-1]

	input_layers = [layer for layer in model.layers_.values()
	            if isinstance(layer, InputLayer)]

	func = get_output([last_hidden, output], None, deterministic=True)

	X_inputs = [theano.Param(input_layer.input_var, name=input_layer.name)
	                    for input_layer in input_layers]

	return theano.function(
            inputs=X_inputs,
            outputs=func,
            )

def get_activations(X, batch_iterator, func):
    activations = []

    for Xb, yb in batch_iterator(X):
        activations.append(NeuralNet.apply_batch_func(func, Xb))

    return activations

def predict(model_id, raw, validation, train):
	#params.DISABLE_CUDNN = True
	params.MULTIPROCESS = False

	d = importlib.import_module("nets.net_" + model_id)
	model, X, y = d.define_net()
	model.load_params_from(params.SAVE_URL + "/" + model_id + "/best_weights")

	f = get_iter_func(model)

	# Decrease batch size because TTA increases it 16-fold
	# Uses too much memory otherwise
	params.BATCH_SIZE = 16
	n_eyes = 2

	io = ImageIO()
	mean, std = io.load_mean_std()

	if validation or train:
		y = util.load_labels()
	else:
		y = util.load_sample_submission()

	keys = y.index.values

	#model.batch_iterator_predict = TTABatchIterator(keys, params.BATCH_SIZE, std, mean, cv = validation or train)
	tta_bi = TTABatchIterator(keys, params.BATCH_SIZE, std, mean, cv = validation or train, n_eyes = n_eyes)
	print "TTAs per image: %i, augmented batch size: %i" % (tta_bi.ttas, tta_bi.ttas * params.BATCH_SIZE * n_eyes)

	if validation:
		X_test = np.load(params.IMAGE_SOURCE + "/X_valid.npy")
	elif train:
		X_test = np.load(params.IMAGE_SOURCE + "/X_train.npy")
	else:
		X_test = np.arange(y.shape[0])

	padded_batches = ceil(X_test.shape[0]/float(params.BATCH_SIZE))

	pred = get_activations(X_test, tta_bi, f)
	
	concat_preds = []

	for batch_pred in pred:
		hidden = batch_pred[0]
		output = batch_pred[1]

		concat = np.concatenate([output, hidden], axis = 1)

		concat_preds.append(concat)

	pred = np.vstack(concat_preds)
	output_units = pred.shape[1]

	#pred = model.predict_proba(X_test)
	pred = pred.reshape(padded_batches, tta_bi.ttas, params.BATCH_SIZE, output_units)
	pred = np.mean(pred, axis = 1)
	pred = pred.reshape(padded_batches * params.BATCH_SIZE, output_units)

	# Remove padded lines
	pred = pred[:X_test.shape[0]]

	# Save unrounded
	#y.loc[keys] = pred

	if validation:
		filename = params.SAVE_URL + "/" + model_id + "/raw_predictions_validation.npy"
	elif train:
		filename = params.SAVE_URL + "/" + model_id + "/raw_predictions_train.npy"
	else:
		filename = params.SAVE_URL + "/" + model_id + "/raw_predictions_test.npy"

	np.save(filename, pred)
	#y.to_csv(filename)
	print "Saved raw predictions to " + filename

	if not raw and not validation and not train:
		W = np.load(params.SAVE_URL + "/" + model_id + "/optimal_thresholds.npy")

		pred = weighted_round(pred, W)

		pred = pred[:, np.newaxis] # add axis for pd compatability

		hist, _ = np.histogram(pred, bins=5)
		print "Distribution over class predictions on test set: ", hist / float(y.shape[0])

		y.loc[keys] = pred

		y.to_csv(params.SAVE_URL + "/" + model_id + "/submission.csv")

		print "Gzipping..."

		if not params.ON_COMA:
			call("gzip -c " + params.SAVE_URL + "/" + model_id + "/submission.csv > " + params.SAVE_URL + "/" + model_id + "/submission.csv.gz", shell=True)

		print "Done! File saved to models/" + model_id + "/submission.csv"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Predict using optimized thresholds and write to file.')
	parser.add_argument('--raw', dest='raw', action='store_true', help = 'ONLY store raw predictions, not rounded')
	parser.add_argument('--train', dest='train', action='store_true', help = 'create predictions for training set, not for test set. automatically sets --raw as well.')
	parser.add_argument('--validation', dest='validation', action='store_true', help = 'create predictions for validation set, not for test set. automatically sets --raw as well.')
	parser.add_argument('model_id', metavar='model_id', type=str, help = 'timestamp ID for the model to optimize')

	args = parser.parse_args()

	assert not (args.train and args.validation)

	predict(args.model_id, args.raw, args.validation, args.train)
