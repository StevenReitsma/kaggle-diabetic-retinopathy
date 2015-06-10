import xgboost as xgb
import numpy as np
import argparse

from params import *
import util

def predict_ensemble(model, activations_test):
	concat_test = np.concatenate(activations_test, axis = 1)

	dtest = xgb.DMatrix(concat_test)

	return model.predict(dtest, ntree_limit = 78)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train ensemble using XGBoost.')
	parser.add_argument('--ensemble', dest='ensemble_model', type=str, nargs='?', required=True, help = 'ensemble model file name.')
	parser.add_argument('model_ids', metavar='model_ids', type=str, nargs='+', help = 'list of models to ensemble.')

	args = parser.parse_args()

	m_test = []

	# Load all models
	for m in args.model_ids:
		t = np.load(params.SAVE_URL + "/" + m + "/raw_predictions_test.csv.npy")
		m_test.append(t)

	model = xgb.Booster(model_file='ensembles/' + args.ensemble_model)

	y = util.load_sample_submission()
	keys = y.index.values

	preds = predict_ensemble(model, m_test)
	preds = np.round(preds[:, np.newaxis])

	y.loc[keys] = preds

	y.to_csv('ensembles/' + args.ensemble_model + '.csv')
