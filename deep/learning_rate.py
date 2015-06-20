import numpy as np
from util import *

class AdjustVariable(object):
	"""
	Adjusts the learning rate according to a set scheme.
	"""
	def __init__(self, name, start=0.03):
		self.name = name
		self.start = start

	# Executed at end of epoch on learning rate and such
	def __call__(self, nn, train_history):
		epoch = train_history[-1]['epoch']

		stop = self.start * 10e-2 * 2
		stop2 = stop * 10e-4 * 2

		ls = np.linspace(self.start, stop, 50)
		ls2 = np.linspace(stop, stop2, nn.max_epochs - 100)

		if epoch <= 50:
			new_value = float32(ls[epoch - 1])
		elif epoch <= 100:
			new_value = float32(ls[-1])
		else:
			new_value = float32(ls2[epoch - 1 - 100])

		getattr(nn, self.name).set_value(new_value)
