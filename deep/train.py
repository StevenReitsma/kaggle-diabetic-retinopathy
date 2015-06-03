# Before doing anything, start Agg backend
import matplotlib as mpl
mpl.use('Agg', force=True) # sets mpl so it doesn't require $DISPLAY on coma

import numpy as np
from params import *

from plotta import PlottaDiabetic, PlottaStart, PlottaUpdate, PlottaStop

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

    # Add plotta
    plotta = PlottaDiabetic(module, "quizoo.nl", 1225)
    net.on_training_started.append(PlottaStart(plotta))
    net.on_training_finished.append(PlottaStop(plotta))
    net.on_epoch_finished.append(PlottaUpdate(plotta))

    net.fit(X, y)

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
