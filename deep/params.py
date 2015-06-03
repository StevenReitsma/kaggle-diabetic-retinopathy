import time
import socket
from multiprocessing import cpu_count

class Params():
	def __init__(self):
		# Check whether we are working on the COMA-cluster
		if "coma" in socket.getfqdn():
			self.IMAGE_SOURCE = "/scratch/sreitsma/kaggle-diabetic-retinopathy/processed"
			self.SAVE_URL = "/scratch/sreitsma/kaggle-diabetic-retinopathy/models"
			self.ON_COMA = True
		else:
			self.IMAGE_SOURCE = "../data/processed"
			self.SAVE_URL = "models"
			self.ON_COMA = False

		self.PIXELS = 256

		# Lower than 64 during training messes up something
		self.BATCH_SIZE = 64
		self.START_LEARNING_RATE = 0.01
		self.MOMENTUM = 0.9

		self.CHANNELS = 3
		self.REGRESSION = True

		self.SUBSET = 10000
		self.AUGMENT = True
		self.COLOR_AUGMENTATION = True
		self.NETWORK_INPUT_TYPE = 'RGB'

		self.MODEL_ID = str(int(time.time()))
		self.CIRCULARIZED_MEAN_STD = True

		self.N_PRODUCERS = cpu_count()

		#Multithreads instead if False
		self.MULTIPROCESS = True

		self.COATES_CENTROIDS = 500

		self.AUGMENTATION_PARAMS = {
					'zoom_range': (1.0, 1.0),
					'rotation_range': (0, 360),
					'translation_range': (-5, 5),
					'do_flip': True,
					'hue_range': (-0.1, 0.1),
					'saturation_range': (-0.1, 0.1),
					'value_range': (-0.1, 0.1)
		}



params = Params()
