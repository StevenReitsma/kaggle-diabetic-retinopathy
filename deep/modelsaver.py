import numpy as np

class ModelSaver(object):
	"""
	Saves the model when it's better than the previous epoch.
	"""
	def __init__(self, output):
		self.output = output
		self.best_valid = np.inf

	# Executed at end of epoch
	def __call__(self, nn, train_history):
		epoch = train_history[-1]['epoch']
		valid_loss = train_history[-1]['valid_loss']

		if valid_loss < self.best_valid:
			self.best_valid = valid_loss
			self.best_valid_epoch = epoch
			nn.save_weights_to(self.output + "_best")