import numpy as np
from params import *
from subprocess import call
from iterators import TTABatchIterator
from imageio import ImageIO

def predict(filename):
	model = joblib.load("models/joblib")

	io = ImageIO()
	mean, std = io.load_mean_std()

	y = util.load_sample_submission()
	keys = y.index.values

	predict_iterator = TTABatchIterator(keys, BATCH_SIZE, std, mean)

	for augmented_batches, batch_keys in predict_iterator():
		probs = []
		for batch in augmented_batches:
			pred = model.predict_proba(batch)
			probs.append(pred)
		tta_predictions = np.mean(probs, axis = 0)

		y[batch_keys] = tta_predictions

	y.to_csv('out.csv')

	print "Gzipping..."

	call("gzip -c out.csv > out.csv.gz", shell=True)

	print "Done! File saved to out.csv and out.csv.gz"

if __name__ == "__main__":
	predict()
