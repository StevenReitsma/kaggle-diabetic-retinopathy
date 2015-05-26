kaggle-diabetic-retinopathy
=========

### General
This repository contains the work of the AI for an Eye team for Kaggle's Diabetic Retinopathy Detection competition. This README.md file contains some information on how to run the algorithms correctly.

### Note
Currently unimplemented files:
* `optimize_ensemble.py`
* `predict_ensemble.py`

Rest is fully implemented.

### Pipeline for single model

1. `python train.py` (use train/test splits that are present in the data/processed folder! this is important!)
 
    This will train the model and put the model, kappa plot and best weights in the `models/<MODEL_ID>` folder.

2. `python optimize_threshold.py <MODEL_ID>`

   This needs the model, best weights, validation split and true labels.
   
   It outputs the optimal thresholds for the validation set to `models/<MODEL_ID>/optimal_thresholds`.
3. `python predict.py <MODEL_ID>`

    This needs the test set images, sample submission, model, best weights and optimal thresholds.
    
    It outputs the predictions for the test set to `models/<MODEL_ID>/submission.csv`.

### Pipeline for ensemble
1. `python train.py`  (use train/test splits that are present in the data/processed folder! this is important!)

    This will train the model and put the model, kappa plot and best weights in the `models/<MODEL_ID>` folder. Do this step for all models you wish to include in the ensemble.
2. `python predict.py --validation <MODEL_ID>`

    This predicts the validation set. Do this for each model in the ensemble. Outputs to `models/<MODEL_ID>/raw_predictions_validation.csv`.
3. `python optimize_ensemble.py [<MODEL_IDS>]`

    This computes optimal weights for the various models in the ensemble. Outputs weights to `models/ensemble_<ENSEMBLE_ID>/weights`. It requires the training labels and validation prediction files `raw_predictions_validation.csv` in each `models/<MODEL_ID>` folder.
4. `python predict_ensemble.py --validation <ENSEMBLE_ID> [<MODEL_IDS>]`

    This uses the `raw_predictions_validation.csv` files of each `<MODEL_ID>` in conjuction with the `models/ensemble_<ENSEMBLE_ID>/weights` to compute the raw predictions for the complete ensemble on the validation set. Saved to `models/ensemble_<ENSEMBLE_ID>/raw_ensemble_predictions_validation.npy`.
5. `python optimize_threshold.py --ensemble <ENSEMBLE_ID>`

    This computes the optimal thresholds for the validation set of the ensemble predictions. Outputs the optimal thresholds to `models/ensemble_<ENSEMBLE_ID>/optimal_thresholds`. It needs the validation predictions in `models/ensemble_<ENSEMBLE_ID>/raw_ensemble_predictions_validation.npy` and the training labels.
6. `python predict.py --raw <MODEL_ID>`

    Create raw predictions on test set for each model.
7. `python predict_ensemble.py <ENSEMBLE_ID> [<MODEL_IDS>]`

    Reads in the `raw_predictions.csv` files for each `<MODEL_ID>`, the `models/ensemble_<ENSEMBLE_ID>/weights` and the `models/ensemble_<ENSEMBLE_ID>/optimal_thresholds` to compute the final submission file which is saved to `models/ensemble_<ENSEMBLE_ID>/submission.csv`.