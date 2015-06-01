import time
import socket

# Check whether we are working on the COMA-cluster
if "coma" in socket.getfqdn():
	IMAGE_SOURCE = "/scratch/sreitsma/kaggle-diabetic-retinopathy/processed"
	SAVE_URL = "/scratch/sreitsma/kaggle-diabetic-retinopathy/models"
	ON_COMA = True
else:
	IMAGE_SOURCE = "../data/processed"
	SAVE_URL = "models"
	ON_COMA = False

PIXELS = 256

# Lower than 64 during training messes up something
BATCH_SIZE = 64
START_LEARNING_RATE = 0.01
MOMENTUM = 0.9

CHANNELS = 3
REGRESSION = True

SUBSET = 0
AUGMENT = True
COLOR_AUGMENTATION = True
NETWORK_INPUT_TYPE = 'RGB'

CIRCULARIZED_MEAN_STD = True
MODEL_ID = str(int(time.time()))
CONCURRENT_AUGMENTATION = False

AUGMENTATION_PARAMS = {
			'zoom_range': (1.0, 1.0),
			'rotation_range': (0, 360),
			'translation_range': (-5, 5),
			'do_flip': True,
			'hue_range': (-0.1, 0.1),
			'saturation_range': (-0.1, 0.1),
			'value_range': (-0.1, 0.1)
}
