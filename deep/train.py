# Before doing anything, start Agg backend
import matplotlib as mpl
mpl.use('Agg', force=True) # sets mpl so it doesn't require $DISPLAY on coma

import numpy as np
from params import *

def fit(model_id = None):
    if model_id is not None:
        params.MODEL_ID = model_id
        custom_id = True
    else:
        custom_id = False

    print "JOB: %s" % params.MODEL_ID

    # Create working directory
    if not os.path.exists(params.SAVE_URL + "/" + params.MODEL_ID):
        os.makedirs(params.SAVE_URL + "/" + params.MODEL_ID)

    if custom_id:
        module = params.MODEL_ID
    else:
        module = "default"

    d = importlib.import_module("nets.net_" + module)

    net, X, y = d.define_net()

    net.fit(X, y)

# Imports are necessary for scoop as Theano gets confused if approached from 4 threads
if __name__ == "__main__":
    import os
    import sys
    import importlib

    # Fix seed
    np.random.seed(42)

    # Get argument
    if len(sys.argv) > 1:
        fit(sys.argv[1])
    else:
        fit()