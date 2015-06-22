kaggle-diabetic-retinopathy
=========

### General
This repository contains the work of the AI for an Eye team for Kaggle's Diabetic Retinopathy Detection competition. This README.md file contains some information on how to run the algorithms correctly.

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
3. `python predict.py --train <MODEL_ID>`
4. `python ensemble_train.py <OUTPUT> [<MODEL_IDS>]`

    This computes optimal weights for the various models in the ensemble. Outputs weights to `ensembles/<OUTPUT>`. It requires the training labels and validation prediction files `raw_predictions_validation.csv` in each `models/<MODEL_ID>` folder.
5. `python predict.py --raw <MODEL_ID>`
6. `python ensemble_predict.py <ENSEMBLE_ID> [<MODEL_IDS>]`

    This uses the `raw_predictions_validation.csv` files of each `<MODEL_ID>` in conjuction with the `ensembles/<ENSEMBLE_ID>` to compute the raw predictions for the complete ensemble on the validation set. Saved to `ensembles/<ENSEMBLE_ID>.csv`.
