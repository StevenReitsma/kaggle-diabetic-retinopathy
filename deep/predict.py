import numpy as np
from params import *
from subprocess import call
from iterators import TTABatchIterator
from imageio import ImageIO
import util
from math import ceil
import argparse
import importlib

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

def predict(model_id, raw, validation, train):
	d = importlib.import_module("nets.net_" + model_id)
	model, X, y = d.define_net()
	model.load_params_from(params.SAVE_URL + "/" + model_id + "/best_weights")

	# Decrease batch size because TTA increases it 16-fold
	# Uses too much memory otherwise
	params.BATCH_SIZE = 32

	io = ImageIO()
	mean, std = io.load_mean_std()

	if validation or train:
		y = util.load_labels()
	else:
		y = util.load_sample_submission()

	keys = y.index.values

	model.batch_iterator_predict = TTABatchIterator(keys, params.BATCH_SIZE, std, mean)
	print "TTAs per image: %i, augmented batch size: %i" % (model.batch_iterator_predict.ttas, model.batch_iterator_predict.ttas * params.BATCH_SIZE)

	if validation:
		X_test = np.load(params.IMAGE_SOURCE + "/X_valid.npy")
	elif train:
		X_test = np.load(params.IMAGE_SOURCE + "/X_train.npy")
	else:
		X_test = np.arange(y.shape[0])

	padded_batches = ceil(X_test.shape[0]/float(params.BATCH_SIZE))

	pred = model.predict_proba(X_test)
	pred = pred.reshape(padded_batches, model.batch_iterator_predict.ttas, params.BATCH_SIZE)
	pred = np.mean(pred, axis = 1)
	pred = pred.reshape(padded_batches * params.BATCH_SIZE)

	# Remove padded lines
	pred = pred[:X_test.shape[0]]

	# Save unrounded
	y.loc[keys] = pred[:, np.newaxis] # add axis for pd compatability

	if validation:
		filename = params.SAVE_URL + "/" + model_id + "/raw_predictions_validation.csv"
	elif train:
		filename = params.SAVE_URL + "/" + model_id + "/raw_predictions_train.csv"
	else:
		filename = params.SAVE_URL + "/" + model_id + "/raw_predictions_test.csv"

	y.to_csv(filename)
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
	parser.add_argument('--train', dest='train', aciton='store_true', help = 'create predictions for training set, not for test set. automatically sets --raw as well.')
	parser.add_argument('--validation', dest='validation', action='store_true', help = 'create predictions for validation set, not for test set. automatically sets --raw as well.')
	parser.add_argument('model_id', metavar='model_id', type=str, help = 'timestamp ID for the model to optimize')

	args = parser.parse_args()

	assert not (args.train and args.validation)

	predict(args.model_id, args.raw, args.validation, args.train)
