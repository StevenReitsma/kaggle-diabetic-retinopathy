import numpy as np
import scipy.optimize
import joblib
from imageio import ImageIO
import util
from iterators import TTABatchIterator
from params import *
from math import *
import argparse
from skll.metrics import kappa

# Define so that it can be pickled
def quadratic_kappa(true, predicted):
    return kappa(true, predicted, weights='quadratic')

def load_validation_set(model_id):
	return np.load("models/" + model_id + "/X_valid.npy")

def load_labels(model_id):
	return np.load("models/" + model_id + "/y_valid.npy")

def load_validation_predictions(filename):
	return np.load(filename)

def compute_validation_predictions(model, weights, validation_set):
	model = joblib.load(model)
	model.load_weights_from(weights)

	io = ImageIO()
	mean, std = io.load_mean_std()

    # Read training labels for the keys
	y = util.load_labels()
	keys = y.index.values

	model.batch_iterator_predict = TTABatchIterator(keys, params.BATCH_SIZE, std, mean, cv = True)
	print "TTAs per image: %i, augmented batch size: %i" % (model.batch_iterator_predict.ttas, model.batch_iterator_predict.ttas * BATCH_SIZE)

	padded_batches = ceil(validation_set.shape[0]/float(params.BATCH_SIZE))

	pred = model.predict_proba(validation_set)
	pred = pred.reshape(padded_batches, model.batch_iterator_predict.ttas, BATCH_SIZE)
	pred = np.mean(pred, axis = 1)
	pred = pred.reshape(padded_batches * BATCH_SIZE)

	# Remove padded lines
	pred = pred[:validation_set.shape[0]]

	return pred

def optimize_thresholds(validation_predictions, true_labels, bins):
	def f(W):
		dim = validation_predictions.reshape((validation_predictions.shape[0], 1))
		rep = dim.repeat(bins, axis = 1)
		delta = W - rep

		# Remove negatives
		delta[(delta < 0)] = np.inf

		preds = np.argmin(delta, axis = 1)

		#mse = np.sum((true_labels - preds)**2)
		kappa = quadratic_kappa(true_labels, preds)

		return 10**(-kappa)

	w_init = np.arange(bins) + 0.5

	initial = f(w_init)

	# Use basinhopping because we're dealing with a highly non-continuous function
	out = scipy.optimize.basinhopping(f, w_init, minimizer_kwargs = {"options": {"disp": True}}, stepsize = 0.1, T = 0.01, niter=10000, niter_success = 2500)

	return out, initial

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Optimize thresholds for a regression problem.')
	parser.add_argument('--subset', dest='subset', action='store_true', help = 'whether to run the optimizer on a subset of the validation set. only works if --fromfile is not set.')
	parser.add_argument('--ensemble', dest='ensemble', action='store_true', help = 'whether to read the predictions from an ensemble. if used, the model_id argument becomes the ensemble_id')
	parser.add_argument('model_id', metavar='model_id', type=str, help = 'timestamp ID for the model to optimize')

	args = parser.parse_args()

	true_labels = load_labels(args.model_id)

	if args.ensemble:
		validation_predictions = load_validation_predictions("models/ensemble_" + args.model_id + "/raw_ensemble_predictions_validation.npy")
	else:
		validation_set = load_validation_set(args.model_id)

		if args.subset:
			validation_set = validation_set[:128]
			true_labels = true_labels[:128]

		validation_predictions = compute_validation_predictions(model = "models/" + args.model_id + "/model", weights = "models/" + args.model_id + "/best_weights", validation_set = validation_set)
	
	thresholds, initial = optimize_thresholds(validation_predictions, true_labels, bins = 5)

	print thresholds

	initial = -np.log10(initial)
	new_kappa = -np.log10(thresholds.fun)

	if args.ensemble:
		np.save("models/ensemble_" + args.model_id + "/optimal_thresholds", thresholds.x)
	else:
		np.save("models/" + args.model_id + "/optimal_thresholds", thresholds.x)

	print "Initial kappa %.5f. Optimized kappa: %.5f. Delta: %.5f" % (initial, new_kappa, new_kappa - initial)
