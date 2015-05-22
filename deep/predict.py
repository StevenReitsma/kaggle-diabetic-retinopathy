import numpy as np
from params import *
from subprocess import call
from iterators import TTABatchIterator
from imageio import ImageIO
import joblib
import util
from math import ceil

# Define so that it can be pickled
def quadratic_kappa(true, predicted):
    return kappa(true, predicted, weights='quadratic')

def predict():
	model = joblib.load("models/joblib")
	model.load_weights_from("models/weights_augm_rot_transl")

	io = ImageIO()
	mean, std = io.load_mean_std()

	y = util.load_sample_submission()

	keys = y.index.values

	model.batch_iterator_predict = TTABatchIterator(keys, BATCH_SIZE, std, mean)

	X_test = np.arange(y.shape[0])
	padded_batches = ceil(y.shape[0]/float(BATCH_SIZE))

	pred = model.predict_proba(X_test)
	pred = pred.reshape(padded_batches, model.batch_iterator_predict.ttas, BATCH_SIZE)
	pred = np.mean(pred, axis = 1)
	pred = pred.reshape(padded_batches * BATCH_SIZE)

	# Remove padded lines
	pred = pred[:y.shape[0]]

	pred = np.minimum(4, np.maximum(0, np.round(pred)))
	pred = pred[:, np.newaxis] # add axis for pd compatability

	hist, _ = np.histogram(pred, bins=5)
	print "Distribution over class predictions on test set: ", hist / float(y.shape[0])

	y.loc[keys] = pred

	y.to_csv('out.csv')

	print "Gzipping..."

	call("gzip -c out.csv > out.csv.gz", shell=True)

	print "Done! File saved to out.csv and out.csv.gz"

if __name__ == "__main__":
	predict()
