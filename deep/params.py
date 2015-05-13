IMAGE_SOURCE = "../data/processed"
IM2BIN_OUTPUT = "../data/processed/images.hdf5"

PIXELS = 256
USE_GPU = True

BATCH_SIZE = 128
START_LEARNING_RATE = 0.01
MOMENTUM = 0.9

CHANNELS = 3
REGRESSION = True

AUGMENTATION_PARAMS = {
			'zoom_range': (1.0, 1.0),
			'rotation_range': (0, 360),
			'shear_range': (0, 0),
			'translation_range': (-5, 5),
			'do_flip': True
		}
