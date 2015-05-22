IMAGE_SOURCE = "../data/processed"

PIXELS = 256
USE_GPU = True

BATCH_SIZE = 32
START_LEARNING_RATE = 0.001
MOMENTUM = 0.9

CHANNELS = 3
REGRESSION = True

SUBSET = 0
AUGMENT = False

AUGMENTATION_PARAMS = {
			'zoom_range': (1.0, 1.0),
			'rotation_range': (0, 360),
			'translation_range': (-5, 5),
			'do_flip': True
}
