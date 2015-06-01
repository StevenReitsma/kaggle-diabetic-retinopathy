import matplotlib.pyplot as plt
from params import *

class Stat(object):
	"""
	Stops training when the validation loss has not decreased for `patience` epochs.
	"""
	def __init__(self):
		pass

	def __call__(self, nn, train_history):
		kappa = [x['kappa'] for x in train_history]

		fig, ax = plt.subplots(1)
		ax.plot(kappa, antialiased=True)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Kappa")

		plt.savefig(params.SAVE_URL + "/" + params.MODEL_ID + "/kappa.png")
		plt.close("all")