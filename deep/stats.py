import matplotlib.pyplot as plt
import time

class Stat(object):
	"""
	Stops training when the validation loss has not decreased for `patience` epochs.
	"""
	def __init__(self):
		self.ts = time.time()

	def __call__(self, nn, train_history):
		kappa = [x['kappa'] for x in train_history]

		#plt.clear()
		fig, ax = plt.subplots(1)
		ax.plot(kappa, antialiased=True)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Kappa")

		plt.savefig('kappa' + str(int(self.ts)) + '.png')
		plt.close("all")