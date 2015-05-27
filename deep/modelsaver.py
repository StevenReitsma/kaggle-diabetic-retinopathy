import numpy as np
import joblib
from params import *

class ModelSaver(object):
	"""
	Saves the model when it's better than the previous epoch.
	"""
	def __init__(self):
		self.best_valid = np.inf

	# Executed at end of epoch
	def __call__(self, nn, train_history):
		epoch = train_history[-1]['epoch']
		valid_loss = train_history[-1]['kappa']

		if valid_loss > self.best_valid:
			self.best_valid = valid_loss
			self.best_valid_epoch = epoch
			nn.save_weights_to("models/" + MODEL_ID + "/best_weights")
			joblib.dump(nn, "models/" + MODEL_ID + "/model")
