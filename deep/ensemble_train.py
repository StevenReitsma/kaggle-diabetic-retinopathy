import xgboost as xgb
import numpy as np
import argparse
from skll.metrics import kappa

from params import *

def bilateralize(data):
	res = data.reshape(data.shape[0] / 2, 2, data.shape[1])
	first = np.concatenate((res[:, 0, :], res[:, 1, :]), axis = 1)
	second = np.concatenate((res[:, 1, :], res[:, 0, :]), axis = 1)

	con = np.dstack((first, second)).transpose(0, 2, 1)
	return con.reshape(data.shape[0], data.shape[1] * 2)

def train_ensemble(activations_train, activations_valid, labels_train, labels_valid, filename, noeval, bilateral):
	# Concatenate the activations of all models together
	concat_train = np.concatenate(activations_train, axis = 1)
	concat_valid = np.concatenate(activations_valid, axis = 1)

	if bilateral:
		# Put last from valid at start of train
		# Our split of training/validation wasn't on an even number, so this is needed for the bilateral information
		# Gives some very minor leakage
		concat_train = np.concatenate(([concat_valid[-1, :]], concat_train), axis = 0)
		concat_valid = concat_valid[:-1, :]

		# Prepare bilateral information
		concat_train = bilateralize(concat_train)
		concat_valid = bilateralize(concat_valid)

	if noeval:
		concat = np.concatenate([concat_valid, concat_train], axis = 0)
		labels_concat = np.concatenate([labels_valid, labels_train], axis = 0)
		dtrain = xgb.DMatrix(concat, labels_concat)
		evals = [(dtrain, 'train')]
	else:
		# Convert to DMatrix for XGBoost
		dtrain = xgb.DMatrix(concat_train, labels_train)
		dvalid = xgb.DMatrix(concat_valid, labels_valid)
		evals = [(dtrain, 'train'), (dvalid, 'eval')]
		
	# Define parameters
	param = {
		'eta': 0.03,
		'gamma': 0,
		'max_depth': 6,
		'min_child_weight': 15,
		'max_delta_step': 0,
		'subsample': 0.8,
		'colsample_bytree': 0.5,
		'objective': 'reg:linear',
		'eval_metric': 'rmse',
		'num_class': 1,
		'silent': 1,
	}

	n_iter = 220

	def kappa_metric(preds, dtrain):
	    labels = dtrain.get_label()
	    return 'kappa', -kappa(labels, preds, weights='quadratic')

	bst = xgb.train(param.items(), dtrain, n_iter, evals, early_stopping_rounds = 50, feval = kappa_metric)

	bst.save_model('ensembles/' + filename)
	np.save('ensembles/' + filename + '_best_iteration.npy', bst.best_iteration)

	best_iteration = bst.best_iteration
	best_score = bst.best_score

	print "Best iteration: %i, best score: %.6f" % (best_iteration, best_score)

	return bst, best_iteration

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train ensemble using XGBoost.')
	parser.add_argument('--noeval', dest='noeval', action='store_true', help = "don't evaluate, use entire training and validation set for training.")
	parser.add_argument('--bilateral', dest='bilateral', action='store_true', help = "whether to incorporate bilateral features.")
	parser.add_argument('output', metavar='output', type=str, help = 'file to output the ensemble model to.')
	parser.add_argument('model_ids', metavar='model_ids', type=str, nargs='+', help = 'list of models to ensemble.')

	args = parser.parse_args()

	m_train = []
	m_valid = []

	# Load all models
	for m in args.model_ids:
		t = np.load(params.SAVE_URL + "/" + m + "/raw_predictions_train.npy")
		v = np.load(params.SAVE_URL + "/" + m + "/raw_predictions_validation.npy")

		m_train.append(t)
		m_valid.append(v)

	# These are the same for every model (if trained correctly)
	y_train = np.load(params.IMAGE_SOURCE + "/y_train.npy")
	y_valid = np.load(params.IMAGE_SOURCE + "/y_valid.npy")

	model, best_iteration = train_ensemble(m_train, m_valid, y_train, y_valid, args.output, args.noeval, args.bilateral)
