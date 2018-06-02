# Alzheimer's Disease Detection
# Test 3D Multi-Scale CNN.
# Author: Qixun QU
# Copyleft: MIT Licience

#     ,,,         ,,,
#   ;"   ';     ;'   ",
#   ;  @.ss$$$$$$s.@  ;
#   `s$$$$$$$$$$$$$$$'
#   $$$$$$$$$$$$$$$$$$
#  $$$$P""Y$$$Y""W$$$$$
#  $$$$  p"$$$"q  $$$$$
#  $$$$  .$$$$$.  $$$$'
#   $$$DaU$$O$$DaU$$$'
#    '$$$$'.^.'$$$$'
#       '&$$$$$&'


from __future__ import print_function


import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from add_models import ADDModels
from sklearn.metrics import (log_loss,
                             roc_curve,
                             recall_score,
                             roc_auc_score,
                             precision_score,
                             confusion_matrix)


class ADDTest(object):

    def __init__(self,
                 paras_name,
                 paras_json_path,
                 weights_save_dir,
                 results_save_dir,
                 test_weights="last",
                 pred_trainset=False):
        '''_INIT__

            Set configurations before testing model.

            Inputs:
            -------

            - paras_name: string, name of hyperparameters set,
                          can be found in hyper_paras.json.
            - paras_json_path: string, path of file which provides
                               hyperparamters, "hyper_paras.json"
                               in this project.
            - weights_save_dir: string, directory path where saves
                                trained model.
            - results_save_dir: string, dorectory to save results.
            - test_wrights: string, which weights used to do test,
                            weights from "last" epoch or weights
                            from "best" epoch.
            - pred_trainset: boolean, whether evaluate model on
                             training set, default is False.

        '''

        if not os.path.isdir(weights_save_dir):
            raise IOError("Model directory is not exist.")

        self.paras_name = paras_name
        self.results_save_dir = results_save_dir
        self.weights = test_weights
        self.pred_trainset = pred_trainset

        # Load hyperparameters
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._load_paras()

        self.weights_path = os.path.join(weights_save_dir,
                                         paras_name, test_weights + ".h5")
        self.results_dir = os.path.join(results_save_dir, paras_name)
        self.create_dir(self.results_dir, rm=False)

        return

    def _load_paras(self):
        '''_LOAD_PARAS

            Load hyperparameters from hyper_paras.json.

        '''

        self.model_name = self.paras["model_name"]
        self.batch_size = self.paras["batch_size"]
        self.scale = self.paras["scale"]
        return

    def _load_model(self):
        '''_LOAD_MODEL

            Create 3D Multi-Scale CNN.

        '''

        self.model = ADDModels(model_name=self.model_name,
                               scale=self.scale).model
        return

    def _pred_evaluate(self, x, y, dataset):
        '''_PRED_EVALUATE

            Predict input data and evaluate performance, including:
            - Accuracy.  ----------|
            - Log loss.  ----------|
            - Precision.  ---------|--> *_*_res.csv
            - Recall.  ------------|
            - ROC AUC.  -----------|
            - Confusion matrix.  --|
            - ROC curve.  ------------> *_*_roc_curve.npy

            Inputs:
            -------

            - x: numpy ndarray, input images.
            - y: numpy ndarray, ground truth labels
            - dataset: string, indicates which set to use,
                       "train", "valid" or "test".

            Outputs:
            --------

            - [dataset]_[self.weights]_res.csv
            - [dataset]_[self.weights]_roc_curve.npy

        '''

        # Helper function to compute metrics
        # true_y: ground truth labels
        # pred_y: predicted labels
        def acc(true_y, pred_y):
            return (true_y == pred_y).all(axis=1).mean()

        def loss(true_y, pred_y):
            return log_loss(true_y, pred_y, normalize=True)

        def precision(true_y, pred_y, label):
            return precision_score(true_y, pred_y, pos_label=label)

        def recall(true_y, pred_y, label):
            return recall_score(true_y, pred_y, pos_label=label)

        print("Dataset to be predicted: " + dataset)

        # Obtain predictions of input data
        pred = self.model.predict(x, self.batch_size, 0)

        # Ground truth labels
        arg_y = np.argmax(y, axis=1)
        arg_y = np.reshape(arg_y, (-1, 1))

        # Indices for AD and NC subjects
        ad = np.where(arg_y == 1)[0]
        nc = np.where(arg_y == 0)[0]

        # Predicted labels
        arg_pred = np.argmax(pred, axis=1)
        arg_pred = np.reshape(arg_pred, (-1, 1))

        # Generate ROC curve
        roc_line = roc_curve(arg_y, pred[:, 1], pos_label=1)
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(arg_y, arg_pred).ravel()

        # A dictionary conains all result to be written
        # in [dataset]_[self.weights]_res.csv
        results = {"name": self.paras_name,
                   "acc": acc(arg_y, arg_pred),
                   "ad_acc": acc(arg_y[ad], arg_pred[ad]),
                   "nc_acc": acc(arg_y[nc], arg_pred[nc]),
                   "loss": loss(y, pred),
                   "ad_loss": loss(y[ad], pred[ad]),
                   "nc_loss": loss(y[nc], pred[nc]),
                   "ad_precision": precision(arg_y, arg_pred, 1),
                   "nc_precision": precision(arg_y, arg_pred, 0),
                   "ad_recall": recall(arg_y, arg_pred, 1),
                   "nc_recall": recall(arg_y, arg_pred, 0),
                   "roc_auc": roc_auc_score(arg_y, pred[:, 1]),
                   "tn": tn, "fp": fp, "fn": fn, "tp": tp}

        # Create pandas DataFrame, and reorder columns
        res_df = pd.DataFrame(data=results, index=[0])
        res_df = res_df[["name", "acc", "ad_acc", "nc_acc",
                         "loss", "ad_loss", "nc_loss",
                         "ad_precision", "ad_recall",
                         "nc_precision", "nc_recall",
                         "roc_auc", "tn", "fp", "fn", "tp"]]

        # Save results to [dataset]_[self.weights]_res.csv
        root_name = [dataset, self.weights]
        res_csv_name = "_".join(root_name + ["res.csv"])
        res_csv_path = os.path.join(self.results_dir, res_csv_name)
        res_df.to_csv(res_csv_path, index=False)

        # Save ROC curve to [dataset]_[self.weights]_roc_curve.npy
        roc_line_name = "_".join(root_name + ["roc_curve.npy"])
        roc_line_path = os.path.join(self.results_dir, roc_line_name)
        np.save(roc_line_path, roc_line)

        return

    def run(self, data):
        '''RUN

            Test model using given data.

            Input:
            ------

            - data: an ADDDataset instance, including features and
                    labels of training, validation and testing set.

        '''

        print("\nTesting the model.\n")

        # Load model and weights
        self._load_model()
        self.model.load_weights(self.weights_path)

        if self.pred_trainset:
            # Predict and evluate on training set
            self._pred_evaluate(data.train_x, data.train_y, "train")

        # Predict and evluate on validation set
        self._pred_evaluate(data.valid_x, data.valid_y, "valid")
        # Predict and evluate on testing set
        self._pred_evaluate(data.test_x, data.test_y, "test")

        return

    @staticmethod
    def load_paras(paras_json_path, paras_name):
        '''LOAD_PARAS

            Load heperparameters from json file.
            See hyper_paras.json.

            Inputs:
            -------

            - paras_name: string, name of hyperparameters set,
                          can be found in hyper_paras.json.
            - paras_json_path: string, path of file which provides
                               hyperparamters, "hyper_paras.json"
                               in this project.

            Output:
            -------

            - A dictionay pf hyperparameters.

        '''

        paras = json.load(open(paras_json_path))
        return paras[paras_name]

    @staticmethod
    def create_dir(dir_path, rm=True):
        '''CREATE_DIR

            Create directory.

            Inputs:
            -------

            - dir_path: string, path of new directory.
            - rm: boolean, remove existing directory or not.

        '''

        if os.path.isdir(dir_path):
            if rm:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
        else:
            os.makedirs(dir_path)
        return


def main(hyper_paras_name, volume_type):
    '''MAIN

        Main process to test model.

        Inputs:
        -------

        - hyper_paras_name: string, the name of hyperparanters set,
                            which can be found in hyper-paras.json.
        - volume_type: string, one of ["whole", "gm", "wm", "csf"].

    '''

    from add_dataset import ADDDataset

    # Basic settings in pre-paras.json, including
    # 1. directory paths for input and output
    # 2. necessary information for splitting dataset
    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    # Get root path of input data
    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, pre_paras["data_dir"])

    # Set directories of input images
    ad_dir = os.path.join(data_dir, pre_paras["ad_in"])
    nc_dir = os.path.join(data_dir, pre_paras["nc_in"])

    # Set directory to save weights and results
    weights_save_dir = os.path.join(parent_dir, pre_paras["weights_save_dir"], volume_type)
    results_save_dir = os.path.join(parent_dir, pre_paras["results_save_dir"], volume_type)

    # Getting splitted dataset
    data = ADDDataset(ad_dir, nc_dir,
                      subj_separated=pre_paras["subj_separated"],
                      volume_type=volume_type,
                      train_prop=pre_paras["train_prop"],
                      valid_prop=pre_paras["valid_prop"],
                      random_state=pre_paras["random_state"],
                      pre_trainset_path=pre_paras["pre_trainset_path"],
                      pre_validset_path=pre_paras["pre_validset_path"],
                      pre_testset_path=pre_paras["pre_testset_path"],
                      data_format=pre_paras["data_format"])
    data.run(pre_split=pre_paras["pre_split"],
             save_split=pre_paras["save_split"],
             save_split_dir=pre_paras["save_split_dir"])

    # Test the model
    test = ADDTest(paras_name=hyper_paras_name,
                   paras_json_path=pre_paras["hyper_paras_json_path"],
                   weights_save_dir=weights_save_dir,
                   results_save_dir=results_save_dir,
                   test_weights=pre_paras["test_weights"],
                   pred_trainset=pre_paras["pred_trainset"])
    test.run(data)

    return


if __name__ == "__main__":

    # Command line
    # python add_test.py --paras=paras-1 --volume=whole

    parser = argparse.ArgumentParser()

    # Set json file path to extract hyperparameters
    help_str = "Select a set of hyper-parameters in hyper_paras.json."
    parser.add_argument("--paras", action="store", default="paras-1",
                        dest="hyper_paras_name", help=help_str)

    # Set type of brain tissue
    help_str = "Select a volume type in ['whole', 'gm', 'wm', 'csf']."
    parser.add_argument("--volume", action="store", default="whole",
                        dest="volume_type", help=help_str)

    args = parser.parse_args()
    main(args.hyper_paras_name, args.volume_type)
