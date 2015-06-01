import numpy as np
from params import *
from subprocess import call
from iterators import TTABatchIterator
from imageio import ImageIO
import joblib
import util
from math import ceil
import argparse

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

def predict(model_id, raw, validation):
	model = joblib.load(SAVE_URL + "/" + model_id + "/model")
	model.load_weights_from(SAVE_URL + "/" + model_id + "/best_weights")

	io = ImageIO()
	mean, std = io.load_mean_std()

	if validation:
		y = util.load_labels()
	else:
		y = util.load_sample_submission()

	keys = y.index.values

	model.batch_iterator_predict = TTABatchIterator(keys, BATCH_SIZE, std, mean)
	print "TTAs per image: %i, augmented batch size: %i" % (model.batch_iterator_predict.ttas, model.batch_iterator_predict.ttas * BATCH_SIZE)

	if validation:
		X_test = np.load(IMAGE_SOURCE + "/X_valid.npy")
	else:
		X_test = np.arange(y.shape[0])

	padded_batches = ceil(X_test.shape[0]/float(BATCH_SIZE))

	pred = model.predict_proba(X_test)
	pred = pred.reshape(padded_batches, model.batch_iterator_predict.ttas, BATCH_SIZE)
	pred = np.mean(pred, axis = 1)
	pred = pred.reshape(padded_batches * BATCH_SIZE)

	# Remove padded lines
	pred = pred[:X_test.shape[0]]

	# Save unrounded
	y.loc[keys] = pred[:, np.newaxis] # add axis for pd compatability

	if validation:
		filename = SAVE_URL + "/" + model_id + "/raw_predictions_validation.csv"
	else:
		filename = SAVE_URL + "/" + model_id + "/raw_predictions.csv"

	y.to_csv(filename)
	print "Saved raw predictions to " + filename

	if not raw and not validation:
		W = np.load(SAVE_URL + "/" + model_id + "/optimal_thresholds.npy")

		pred = weighted_round(pred, W)

		pred = pred[:, np.newaxis] # add axis for pd compatability

		hist, _ = np.histogram(pred, bins=5)
		print "Distribution over class predictions on test set: ", hist / float(y.shape[0])

		y.loc[keys] = pred

		y.to_csv(SAVE_URL + "/" + model_id + "/submission.csv")

		print "Gzipping..."

		if not ON_COMA:
			call("gzip -c " + SAVE_URL + "/" + model_id + "/submission.csv > " + SAVE_URL + "/" + model_id + "/submission.csv.gz", shell=True)

		print "Done! File saved to models/" + model_id + "/submission.csv"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Predict using optimized thresholds and write to file.')
	parser.add_argument('--raw', dest='raw', action='store_true', help = 'ONLY store raw predictions, not rounded')
	parser.add_argument('--validation', dest='validation', action='store_true', help = 'create predictions for validation set, not for test set. automatically sets --raw as well.')
	parser.add_argument('model_id', metavar='model_id', type=str, help = 'timestamp ID for the model to optimize')

	args = parser.parse_args()

	predict(args.model_id, args.raw, args.validation)
