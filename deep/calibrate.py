import numpy as np
import pandas as pd

REAL_DISTRIBUTION = np.array([0.73478335,  0.06954962,  0.15065763,  0.02485338,  0.02015601])

def calibrate_distribution(input):
	preds = pd.DataFrame.from_csv(input)

	sorted_df = preds.sort(columns = 'level')

	images_per_bin = np.cast['int32'](np.round(REAL_DISTRIBUTION * sorted_df.shape[0]))
	start_indices_per_bin = np.concatenate(([0], np.cumsum(images_per_bin)))

	bins = []

	for i in range(len(REAL_DISTRIBUTION)):
		r = sorted_df.ix[start_indices_per_bin[i]:start_indices_per_bin[i+1]]
		keys = r.index.values

		preds.loc[keys] = i

	preds['level'] = preds['level'].astype(int)
	preds.to_csv(input + '_calibrated.csv')

if __name__ == "__main__":
	calibrate_distribution("ensembles/512_moreits.model_unrounded.csv")
	calibrate_optimize("ensembles/512_moreits.model_unrounded.csv") #TODO