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
        '''

        if not os.path.isdir(weights_save_dir):
            raise IOError("Model directory is not exist.")

        self.paras_name = paras_name
        self.results_save_dir = results_save_dir
        self.weights = test_weights
        self.pred_trainset = pred_trainset
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._resolve_paras()

        self.weights_path = os.path.join(weights_save_dir,
                                         paras_name, test_weights + ".h5")
        self.results_dir = os.path.join(results_save_dir, paras_name)
        self.create_dir(self.results_dir, rm=False)

        return

    def _resolve_paras(self):
        self.model_name = self.paras["model_name"]
        self.batch_size = self.paras["batch_size"]
        self.scale = self.paras["scale"]
        return

    def _load_model(self):
        self.model = ADDModels(model_name=self.model_name,
                               scale=self.scale).model
        return

    def _pred_evaluate(self, x, y, dataset):

        def acc(true_y, pred_y):
            return (true_y == pred_y).all(axis=1).mean()

        def loss(true_y, pred_y):
            return log_loss(true_y, pred_y, normalize=True)

        def precision(true_y, pred_y, label):
            return precision_score(true_y, pred_y, pos_label=label)

        def recall(true_y, pred_y, label):
            return recall_score(true_y, pred_y, pos_label=label)

        print("Dataset to be predicted: " + dataset)
        pred = self.model.predict(x, self.batch_size, 0)

        arg_y = np.argmax(y, axis=1)
        arg_y = np.reshape(arg_y, (-1, 1))
        ad = np.where(arg_y == 1)[0]
        nc = np.where(arg_y == 0)[0]

        arg_pred = np.argmax(pred, axis=1)
        arg_pred = np.reshape(arg_pred, (-1, 1))

        roc_line = roc_curve(arg_y, pred[:, 1], pos_label=1)
        tn, fp, fn, tp = confusion_matrix(arg_y, arg_pred).ravel()

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

        res_df = pd.DataFrame(data=results, index=[0])
        res_df = res_df[["name", "acc", "ad_acc", "nc_acc",
                         "loss", "ad_loss", "nc_loss",
                         "ad_precision", "ad_recall",
                         "nc_precision", "nc_recall",
                         "roc_auc", "tn", "fp", "fn", "tp"]]

        root_name = [dataset, self.weights]
        res_csv_name = "_".join(root_name + ["res.csv"])
        roc_line_name = "_".join(root_name + ["roc_curve.npy"])

        res_csv_path = os.path.join(self.results_dir, res_csv_name)
        res_df.to_csv(res_csv_path, index=False)

        roc_line_path = os.path.join(self.results_dir, roc_line_name)
        np.save(roc_line_path, roc_line)

        return

    def run(self, data):
        '''RUN
        '''

        print("\nTesting the model.\n")

        self._load_model()
        self.model.load_weights(self.weights_path)

        if self.pred_trainset:
            self._pred_evaluate(data.train_x, data.train_y, "train")

        self._pred_evaluate(data.valid_x, data.valid_y, "valid")
        self._pred_evaluate(data.test_x, data.test_y, "test")

        return

    @staticmethod
    def load_paras(paras_json_path, paras_name):
        paras = json.load(open(paras_json_path))
        return paras[paras_name]

    @staticmethod
    def create_dir(dir_path, rm=True):
        if os.path.isdir(dir_path):
            if rm:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
        else:
            os.makedirs(dir_path)
        return


def main(hyper_paras_name, volume_type):

    from add_dataset import ADDDataset

    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, pre_paras["data_dir"])

    ad_dir = os.path.join(data_dir, pre_paras["ad_in"])
    nc_dir = os.path.join(data_dir, pre_paras["nc_in"])

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

    # Testing the model
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
    help_str = "Select a set of hyper-parameters in hyper_paras.json."
    parser.add_argument("--paras", action="store", default="paras-1",
                        dest="hyper_paras_name", help=help_str)
    help_str = "Select a volume type in ['whole', 'gm', 'wm', 'csf']."
    parser.add_argument("--volume", action="store", default="whole",
                        dest="volume_type", help=help_str)

    args = parser.parse_args()
    main(args.hyper_paras_name, args.volume_type)
