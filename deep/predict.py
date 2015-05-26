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

def predict(model_id):
	model = joblib.load("models/" + model_id + "/model")
	#model.load_weights_from("models/" + model_id + "/best_weights")
	model.load_weights_from("old_files/weights/weights_augm_rot_transl_hsv")

	W = np.load("models/" + model_id + "/optimal_thresholds.npy")

	io = ImageIO()
	mean, std = io.load_mean_std()

	y = util.load_sample_submission()

	keys = y.index.values

	model.batch_iterator_predict = TTABatchIterator(keys, BATCH_SIZE, std, mean)
	print "TTAs per image: %i, augmented batch size: %i" % (model.batch_iterator_predict.ttas, model.batch_iterator_predict.ttas * BATCH_SIZE)

	X_test = np.arange(y.shape[0])
	padded_batches = ceil(y.shape[0]/float(BATCH_SIZE))

	pred = model.predict_proba(X_test)
	pred = pred.reshape(padded_batches, model.batch_iterator_predict.ttas, BATCH_SIZE)
	pred = np.mean(pred, axis = 1)
	pred = pred.reshape(padded_batches * BATCH_SIZE)

	# Remove padded lines
	pred = pred[:y.shape[0]]

	# Save unrounded
	y.loc[keys] = pred[:, np.newaxis] # add axis for pd compatability
	y.to_csv("models/" + model_id + "/raw_predictions.csv")

	pred = weighted_round(pred, W)

	pred = pred[:, np.newaxis] # add axis for pd compatability

	hist, _ = np.histogram(pred, bins=5)
	print "Distribution over class predictions on test set: ", hist / float(y.shape[0])

	y.loc[keys] = pred

	y.to_csv("models/" + model_id + "/submission.csv")

	print "Gzipping..."

	call("gzip -c models/" + model_id + "/submission.csv > models/" + model_id + "/submission.csv.gz", shell=True)

	print "Done! File saved to models/" + model_id + "/submission.csv.gz"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Predict using optimized thresholds and write to file.')
	parser.add_argument('model_id', metavar='model_id', type=str, help = 'timestamp ID for the model to optimize')

	args = parser.parse_args()

	predict(args.model_id)
