# Alzheimer's Disease Detection
# Refine AD detection by
# feature fusion and selection.
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
        '''__INIT__

            Load parameters from feat_paras.json.

            Inputs:
            -------

            - paras_name: string, name of parameters set in
                          rfn_paras.json.
            - paras_json_path: string, path of file which provides
                               parameters. In this case, it is
                               "rfn_paras.json".
            - features_dir: string, path of root directory to save
                            output features.

        '''

        # Load parameters in rfn_paras.json
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._load_paras()

        self.feat_root = features_dir

        return

    def _load_paras(self):
        '''_LOAD_PARAS

            Load parameters for training XGBoost classifier
            and MLP classifier.

        '''

        self.fdir = self.paras["fdir"]                      # subdir in self.feat_root
        self.feat = self.paras["feat"]                      # choose tissues
        self.fnum = self.paras["fnum"]                      # number of features of each tissue
        self.select = self.paras["select"]                  # If true, do feature selection
        self.xgb_paras = self.paras["xgb_paras"]            # parameters for XGBoost
        self.xgb_verbose = self.paras["xgb_fit_verbose"]    # if True, show verbose
        self.threshold = self.paras["feat_threshold"]       # threshold for feature selection
        self.save_fs = self.paras["save_fs_idx_file"]       # if True, save feature indices to txt file
        self.fs_file_ext = self.paras["fs_idx_file_ext"]    # postfix of txt file
        self.mlp_runs_num = self.paras["mlp_runs_num"]      # number of MLP runs
        self.mlp_hl_size = self.paras["mlp_hl_size"]        # hidden layer size of MLP without feature selection
        self.fs_mlp_hl_size = self.paras["fs_mlp_hl_size"]  # hidden layer size of MLP with feature selection
        self.mlp_paras = self.paras["mlp_paras"]            # parameters for MLP

        return

    def run(self, save_features=False, save_features_dir=None):
        '''RUN

            Run refination of AD detection.
            -1- Fuse tissues' features, and input them into MLP (hidden layer size is 256).
            -2- Fuse tissues' features, select features by XGBoost,
                input selected features into MLP (hidden layer size is 64).

            Inputs:
            -------

            - save_features: boolean, if True, save indices of features to txt file.
                             Default is False.
            - save_features_dir: string, path of directory to save txt file of features' indices.
                                 Default is None.

            Output:
            -------

            - If save_features is True, txt file will be saved into given directory.

        '''

        # Load feature files paths of partitions
        trainset, validset, testset = self._load_feat_path()

        # Load feature data of partitions
        paras = {"feat": self.feat, "fnum": self.fnum}
        X_train, y_train = self.get_feat_data(trainset, "train set", **paras)
        X_valid, y_valid = self.get_feat_data(validset, "valid set", **paras)
        X_test, y_test = self.get_feat_data(testset, "test set", **paras)

        if self.select:
            # Apply XGBoost to select features
            print("Feature selection by XGB")
            xgb_clf, fs_idx, importance = self._feature_selection(X_train, y_train,
                                                                  X_valid, y_valid)
            self.evaluate(xgb_clf, X_train, y_train, "Train")
            self.evaluate(xgb_clf, X_valid, y_valid, "Valid")
            self.evaluate(xgb_clf, X_test, y_test, "Test")

            # Extract selected features
            X_train = X_train[:, fs_idx]
            X_valid = X_valid[:, fs_idx]
            X_test = X_test[:, fs_idx]

            if save_features:
                # Save indices of selected features into txt file
                self._save_features(fs_idx, importance, save_features_dir)

        # Set the number of neurons in hidden layer of MLP
        # Without feature selection: 256
        # With feature selection: 64
        self.mlp_paras["hidden_layer_sizes"] = \
            self.fs_mlp_hl_size if self.select else self.mlp_hl_size

        valid_res, test_res = [], []
        for i in range(self.mlp_runs_num):
            print("\nTrain NN - ", i + 1)
            # Train MLP
            nn_clf = self._neural_network(X_train, y_train)

            # Evaluate the performance on validation set and testing set
            self.evaluate(nn_clf, X_train, y_train, "Train")
            valid_acc = self.evaluate(nn_clf, X_valid, y_valid, "Valid")
            test_acc = self.evaluate(nn_clf, X_test, y_test, "Test", True)

            # Save results
            valid_res.append(valid_acc)
            test_res.append(test_acc)

        # Print average results
        print("\nAverage Results:")
        print("Valid ACC: {0:.4f} +/- {1:.4f}".format(np.mean(valid_res), np.std(valid_res)))
        test_res_mean, test_res_std = np.mean(test_res, axis=0), np.std(test_res, axis=0)
        print("Test ACC: {0:.4f} +/- {1:.4f}".format(test_res_mean[0], test_res_std[0]))
        print("Test AD ACC: {0:.4f} +/- {1:.4f}".format(test_res_mean[1], test_res_std[1]))
        print("Test NC ACC: {0:.4f} +/- {1:.4f}".format(test_res_mean[2], test_res_std[2]))

        return

    def _load_feat_path(self):
        '''_LOAD_FEAT_PATH

            Generate feature file paths for training set,
            validation set and testing set.

            Outputs:
            --------

            - trainset, validset, testset: partition information,
              each output is a list with two columns, directory
              of one scan's features and its label.

        '''

        # Load partition infotmation
        feat_dir = os.path.join(self.feat_root, self.fdir)
        trainset = self.get_feat_path(os.path.join(feat_dir, "train"))
        validset = self.get_feat_path(os.path.join(feat_dir, "valid"))
        testset = self.get_feat_path(os.path.join(feat_dir, "test"))

        return trainset, validset, testset

    def _feature_selection(self, X, y, Xv, yv):
        '''_FEATURE_SELECTION

            Apply XGBoost to do feature selection.

            Inputs:
            -------

            - X: numpy ndarray, features of training set.
            - y: numpy ndarray, labels of training set.
            - Xv: numpy ndarray, features of validation set.
            - yv: numpy ndarray, laebls of validation set.

            Outputs:
            --------

            - clf: instance of XGBClassifier, trian model.
            - fs_idx: list, indecies of selected features.
            - importance: list, importance of selected features.

        '''

        # Train XGBoost classifier
        clf = xgb(**self.xgb_paras)
        clf.fit(X, y, eval_set=[(X, y), (Xv, yv)],
                eval_metric="error", verbose=False)

        # Extract indices of important features
        importance = clf.feature_importances_
        fs_idx = np.where(importance > self.threshold)[0]
        importance = importance[fs_idx]
        print("Number of important features: ", len(fs_idx))

        return clf, fs_idx, importance

    def _neural_network(self, X, y):
        '''_NEURAL_NETWORK

            Train MLP to do final classification.

            Inputs:
            -------

            - X: numpy ndarray, features array.
            - y: numpy ndarray, labels list.

            Output:
            -------

            - clf: instandce of MLPClassifier, trained model.

        '''

        # Train MLP
        clf = mlp(**self.mlp_paras)
        clf.fit(X, y)

        # To save trained model
        # import joblib at the begining of this script
        # joblib.dump(clf, "mlp.pkl")

        return clf

    def _save_features(self, fs_idx, importance, save_dir):
        '''_SAVEFEAT_IDX

            Save indices of selected features into txt file.

            Inputs:
            -------

            - fs_idx: list, indices of important features.
            - importance: list, importance of selected features.
            - save_dir: string, path of directory to save file.

            Output:
            -------

            - A text file of indices.

        '''

        if type(self.feat).__name__ == "unicode":
            # self.feat is a string, such as "gm", "wm" or "csf"
            # Conver the string to a list in length 1
            feat_list = [self.feat]
        else:
            feat_list = self.feat

        # Generate text file's name
        feat_idx_name = self.fdir + "_" + "_".join(feat_list) + "_" + \
            str(self.fnum) + self.fs_file_ext
        # Generate file's path
        feat_idx_path = os.path.join(save_dir, feat_idx_name)

        # Write file
        with open(feat_idx_path, "w") as idx_file:
            for idx, ip in zip(fs_idx, importance):
                idx_file.write("{},{}\n".format(idx, ip))

        return

    @staticmethod
    def evaluate(clf, X, y, mode, ad_nc=False):
        '''EVALUATE

            Compute prediction accuracy on dataset
            with given model.

            Inputs:
            -------

            - X: numpy ndarray, features array.
            - y: numpy ndarray, labels list.
            - mode: string, indicates which partition is used.
            - ad_nc: boolean, if True, print accuracy of each class.

            Outputs:
            --------

            If ad_nc is True:
            - A list of overall accuracy, AD accuracy and NC accuracy.
            Else:
            - A float of overall accuracy.

        '''

        # Get predictions of given data
        y_pred = clf.predict(X)

        # Compute overall accuracy
        acc = accuracy_score(y, y_pred)
        print(mode + " Accuracy:", acc)

        if ad_nc:
            # Comput AD and NC accuracy
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
        '''LOAD_PARAS

            Load parameters from json file for refination.
            See rfn_paras.json.

            Inputs:
            -------

            - paras_name: string, name of parameters set,
                          can be found in rfn_paras.json.
            - paras_json_path: string, path of file which provides
                               paramters, "rfn_paras.json".

            Output:
            -------

            - A dictionay pf parameters for refination.

        '''

        paras = json.load(open(paras_json_path))
        return paras[paras_name]

    @staticmethod
    def get_feat_path(feat_dir):
        '''GET_FEAT_PATH

            Generate partition information from given directory.

            Input:
            ------

            - feat_dir: string, path of directory of features of
                        three partitions.

            Output:
            -------

            - feat_info: list with two columns, directory of one
                         scan's feaures and the label of this scan.

        '''

        groups = os.listdir(feat_dir)  # ["AD", "NC"]
        feat_info = []
        for group in groups:
            # Set label to 1 for AD subjects
            # Set label to 0 for NC subjects
            label = 1 if group == "AD" else 0
            group_dir = os.path.join(feat_dir, group)
            for subj in os.listdir(group_dir):
                subj_dir = os.path.join(group_dir, subj)
                for scan in os.listdir(subj_dir):
                    # Directory path of features of one scan
                    scan_dir = os.path.join(subj_dir, scan)
                    feat_info.append([scan_dir, label])

        shuffle(feat_info)
        return feat_info

    @staticmethod
    def get_feat_data(dataset, mode, feat, fnum):
        '''GET_FEAT_DATA

            Load features to numpy array.

            Inputs:
            -------

            - dataset: list with two columns, scan features' directory
                       and label for this scan.
            - mode: string, indicates which partition is used.
            - feat: list or string, features of those tissues are used.
            - fnum: int, the number of features extracted from each tissue.

            Output:
            -------

            - Features array of all scans.
            - Labels array of all scans.

        '''

        if type(feat).__name__ == "list" and len(feat) == 1:
            # If feat is a list and its length is 1
            # such as feat = ["gm"], convert it to a sting "gm"
            feat = feat[0]

        # Postfix of feature file
        postfix = "_" + str(fnum) + ".npy"

        if type(feat).__name__ == "unicode":
            # If feat is a string, means only one tissue are used
            X, y = [], []
            feat_name = feat + postfix
            for subj in dataset:
                # Load features according partition information
                y.append(subj[1])
                feat_path = os.path.join(subj[0], feat_name)
                X.append(np.load(feat_path))
            return np.squeeze(np.array(X), axis=1), np.array(y)
        else:  # feat is a list contains more than one string
            # Such as feat = ["gm", "csf"]
            all_feats, load_y = [], True
            y = []
            for f in feat:
                # Load features of each tissue
                X = []
                feat_name = f + postfix
                for subj in dataset:
                    # Load features and labels
                    if load_y:
                        y.append(subj[1])
                    feat_path = os.path.join(subj[0], feat_name)
                    X.append(np.load(feat_path))
                X = np.squeeze(np.array(X), axis=1)
                if load_y:
                    # Labels are only loaded once
                    y = np.array(y)
                    load_y = False
                all_feats.append(X)
            all_feats = np.concatenate(all_feats, axis=1)
            return all_feats, y

        return


def main(rfn_paras_name):

    # Get basic condiguration in pre_paras.json
    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    # Root directory of features extracted by trained model
    parent_dir = os.path.dirname(os.getcwd())
    features_dir = os.path.join(parent_dir, pre_paras["features_dir"])

    # Set output file path for selected features
    save_features = pre_paras["save_features"]
    save_features_dir = pre_paras["save_features_dir"]

    # path of json file to provide parameters for refination
    rfn_paras_json_path = "rfn_paras.json"

    # Refine AD detection
    rfn = ADDRefine(paras_name=rfn_paras_name,
                    paras_json_path=rfn_paras_json_path,
                    features_dir=features_dir)
    rfn.run(save_features=save_features,
            save_features_dir=save_features_dir)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Set the name of parameters set in rfn_paras.json
    help_str = "Select a set of parameters in rfn_paras.json."
    parser.add_argument("--paras", action="store", default="refine-1",
                        dest="rfn_paras_name", help=help_str)

    args = parser.parse_args()
    main(args.rfn_paras_name)
