import time

IMAGE_SOURCE = "../data/processed"

PIXELS = 256
USE_GPU = True

BATCH_SIZE = 32
START_LEARNING_RATE = 0.01
MOMENTUM = 0.9

CHANNELS = 3
REGRESSION = True

SUBSET = 0
AUGMENT = True
COLOR_AUGMENTATION = True
NETWORK_INPUT_TYPE = 'RGB'

MODEL_ID = str(int(time.time()))

AUGMENTATION_PARAMS = {
			'zoom_range': (1.0, 1.0),
			'rotation_range': (0, 360),
			'translation_range': (-5, 5),
			'do_flip': True,
			'hue_range': (-0.1, 0.1),
			'saturation_range': (-0.1, 0.1),
			'value_range': (-0.1, 0.1)
}
