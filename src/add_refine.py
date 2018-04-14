from __future__ import print_function


import os
import json
import argparse
import numpy as np
from random import shuffle
from xgboost import XGBClassifier as xgb
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier as mlp


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class ADDRefine(object):

    def __init__(self, paras_name,
                 paras_json_path, features_dir):
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._resolve_paras()

        self.feat_root = features_dir

        return

    def _resolve_paras(self):
        self.fdir = self.paras["fdir"]
        self.feat = self.paras["feat"]
        self.fnum = self.paras["fnum"]
        self.select = self.paras["select"]
        self.xgb_paras = self.paras["xgb_paras"]
        self.xgb_verbose = self.paras["xgb_fit_verbose"]
        self.threshold = self.paras["feat_threshold"]
        self.save_fs = self.paras["save_fs_idx_file"]
        self.fs_file_ext = self.paras["fs_idx_file_ext"]
        self.mlp_runs_num = self.paras["mlp_runs_num"]
        self.mlp_hl_size = self.paras["mlp_hl_size"]
        self.fs_mlp_hl_size = self.paras["fs_mlp_hl_size"]
        self.mlp_paras = self.paras["mlp_paras"]
        return

    def run(self, save_features=False, save_features_dir=None):

        trainset, validset, testset = self._load_feat_path()

        paras = {"feat": self.feat, "fnum": self.fnum}
        X_train, y_train = self.get_feat_data(trainset, "train set", **paras)
        X_valid, y_valid = self.get_feat_data(validset, "valid set", **paras)
        X_test, y_test = self.get_feat_data(testset, "test set", **paras)

        if self.select:
            print("Feature selection by XGB")
            xgb_clf, fs_idx = self._feature_selection(X_train, y_train,
                                                      X_valid, y_valid)
            self.evaluate(xgb_clf, X_train, y_train, "Train")
            self.evaluate(xgb_clf, X_valid, y_valid, "Valid")
            self.evaluate(xgb_clf, X_test, y_test, "Test")

            X_train = X_train[:, fs_idx]
            X_valid = X_valid[:, fs_idx]
            X_test = X_test[:, fs_idx]

            if save_features:
                self._save_feat_idx(fs_idx, save_features_dir)

        self.mlp_paras["hidden_layer_sizes"] = \
            self.fs_mlp_hl_size if self.select else self.mlp_hl_size

        valid_res, test_res = [], []
        for i in range(self.mlp_runs_num):
            print("\nTrain NN - ", i + 1)
            nn_clf = self._neural_network(X_train, y_train)

            self.evaluate(nn_clf, X_train, y_train, "Train")
            valid_acc = self.evaluate(nn_clf, X_valid, y_valid, "Valid")
            test_acc = self.evaluate(nn_clf, X_test, y_test, "Test", True)

            valid_res.append(valid_acc)
            test_res.append(test_acc)

        print("\nAverage Results:")
        print("Valid ACC: {0:.4f} +/- {1:.4f}".format(np.mean(valid_res), np.std(valid_res)))
        test_res_mean, test_res_std = np.mean(test_res, axis=0), np.std(test_res, axis=0)
        print("Test ACC: {0:.4f} +/- {1:.4f}".format(test_res_mean[0], test_res_std[0]))
        print("Test AD ACC: {0:.4f} +/- {1:.4f}".format(test_res_mean[1], test_res_std[1]))
        print("Test NC ACC: {0:.4f} +/- {1:.4f}".format(test_res_mean[2], test_res_std[2]))

        return

    def _load_feat_path(self):
        feat_dir = os.path.join(self.feat_root, self.fdir)
        trainset = self.get_feat_path(os.path.join(feat_dir, "train"))
        validset = self.get_feat_path(os.path.join(feat_dir, "valid"))
        testset = self.get_feat_path(os.path.join(feat_dir, "test"))
        return trainset, validset, testset

    def _feature_selection(self, X, y, Xv, yv):
        clf = xgb(**self.xgb_paras)
        clf.fit(X, y, eval_set=[(X, y), (Xv, yv)],
                eval_metric="error", verbose=False)
        importance = clf.feature_importances_
        fs_idx = np.where(importance > self.threshold)[0]
        print("Number of important features: ", len(fs_idx))
        return clf, fs_idx

    def _neural_network(self, X, y):
        clf = mlp(**self.mlp_paras)
        clf.fit(X, y)
        # joblib.dump(clf, "mlp.pkl")
        return clf

    def _save_feat_idx(self, fs_idx, save_dir):
        if type(self.feat).__name__ == "unicode":
            feat_list = [self.feat]
        else:
            feat_list = self.feat
        feat_idx_name = self.fdir + "_" + "_".join(feat_list) + "_" + \
            str(self.fnum) + self.fs_file_ext
        feat_idx_path = os.path.join(save_dir, feat_idx_name)
        with open(feat_idx_path, "w") as idx_file:
            for idx in fs_idx:
                idx_file.write("{}\n".format(idx))
        return

    @staticmethod
    def evaluate(clf, X, y, mode, ad_nc=False):
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        print(mode + " Accuracy:", acc)
        if ad_nc:
            ad_idx = np.where(y == 1)[0]
            nc_idx = np.where(y == 0)[0]
            ad_acc = accuracy_score(y[ad_idx], y_pred[ad_idx])
            nc_acc = accuracy_score(y[nc_idx], y_pred[nc_idx])
            print(mode + " AD Accuracy: ", ad_acc)
            print(mode + " NC Accuracy: ", nc_acc)
            return [acc, ad_acc, nc_acc]
        else:
            return acc

    @staticmethod
    def load_paras(paras_json_path, paras_name):
        paras = json.load(open(paras_json_path))
        return paras[paras_name]

    @staticmethod
    def get_feat_path(feat_dir):
        groups = os.listdir(feat_dir)
        feat_info = []
        for group in groups:
            label = 1 if group == "AD" else 0
            group_dir = os.path.join(feat_dir, group)
            for subj in os.listdir(group_dir):
                subj_dir = os.path.join(group_dir, subj)
                for scan in os.listdir(subj_dir):
                    scan_dir = os.path.join(subj_dir, scan)
                    feat_info.append([scan_dir, label])

        shuffle(feat_info)
        return feat_info

    @staticmethod
    def get_feat_data(dataset, mode, feat, fnum):

        if type(feat).__name__ == "list" and len(feat) == 1:
            feat = feat[0]

        postfix = "_" + str(fnum) + ".npy"
        if type(feat).__name__ == "unicode":
            X, y = [], []
            feat_name = feat + postfix
            for subj in dataset:
                y.append(subj[1])
                feat_path = os.path.join(subj[0], feat_name)
                X.append(np.load(feat_path))
            return np.squeeze(np.array(X), axis=1), np.array(y)
        else:  # feat is a list contains more than one string
            all_feats, load_y = [], True
            y = []
            for f in feat:
                X = []
                feat_name = f + postfix
                for subj in dataset:
                    if load_y:
                        y.append(subj[1])
                    feat_path = os.path.join(subj[0], feat_name)
                    X.append(np.load(feat_path))
                X = np.squeeze(np.array(X), axis=1)
                if load_y:
                    y = np.array(y)
                    load_y = False
                all_feats.append(X)
            all_feats = np.concatenate(all_feats, axis=1)
            return all_feats, y

        return


def main(rfn_paras_name):

    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    parent_dir = os.path.dirname(os.getcwd())
    features_dir = os.path.join(parent_dir, pre_paras["features_dir"])
    save_features = pre_paras["save_features"]
    save_features_dir = pre_paras["save_features_dir"]

    rfn_paras_json_path = "rfn_paras.json"

    rfn = ADDRefine(paras_name=rfn_paras_name,
                    paras_json_path=rfn_paras_json_path,
                    features_dir=features_dir)
    rfn.run(save_features=save_features,
            save_features_dir=save_features_dir)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    help_str = "Select a set of parameters in rfn_paras.json."
    parser.add_argument("--paras", action="store", default="refine-1",
                        dest="rfn_paras_name", help=help_str)

    args = parser.parse_args()
    main(args.rfn_paras_name)
