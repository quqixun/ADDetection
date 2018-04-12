from __future__ import print_function

import os
import numpy as np
from xgboost import XGBClassifier as xgb
from xgboost import plot_importance
from random import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_sepdata_path(data_dir):
    groups = os.listdir(data_dir)
    data_info = []
    for group in groups:
        if group == "AD":
            label = 1
        else:
            label = 0

        group_dir = os.path.join(data_dir, group)
        for subj in os.listdir(group_dir):
            subj_dir = os.path.join(group_dir, subj)
            for scan in os.listdir(subj_dir):
                scan_dir = os.path.join(subj_dir, scan)
                data_info.append([scan_dir, label])

    shuffle(data_info)
    return data_info


def load_data(info, mode, feat, fnum):
    gm, wm, csf, whole, y = [], [], [], [], []
    print("Loading {} data ...".format(mode))
    for subject in info:
        dir_path = subject[0]
        label = subject[1]
        y.append(label)

        gm.append(np.load(os.path.join(dir_path, "gm_" + str(fnum) + ".npy")))
        wm.append(np.load(os.path.join(dir_path, "wm_" + str(fnum) + ".npy")))
        csf.append(np.load(os.path.join(dir_path, "csf_" + str(fnum) + ".npy")))

        if feat == "whole":
            whole.append(np.load(os.path.join(dir_path, "whole_" + str(fnum) + ".npy")))

    fnum = gm[0].shape[1]
    gm = np.squeeze(np.array(gm), axis=1)
    wm = np.squeeze(np.array(wm), axis=1)
    csf = np.squeeze(np.array(csf), axis=1)
    if feat == "whole":
        whole = np.squeeze(np.array(whole), axis=1)
    # allf = np.concatenate([gm, wm, csf], axis=1)
    allf = np.concatenate([gm, csf], axis=1)

    if feat == "gm":
        X = gm
    elif feat == "wm":
        X = wm
    elif feat == "csf":
        X = csf
    elif feat == "whole":
        X = whole
    elif feat == "all":
        X = allf

    y = np.array(y)
    # y[y == 0] = -1

    return X, y


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    # data_dir = os.path.join(parent_dir, "features_old", "pre_pve")
    # data_dir = os.path.join(parent_dir, "features", "pre")
    data_dir = os.path.join(parent_dir, "features", "cgwpre")
    # data_dir = os.path.join(parent_dir, "features", "cgwnew")
    # data_dir = os.path.join(parent_dir, "features", "cgwbest")
    trainset_info = get_sepdata_path(os.path.join(data_dir, "train"))
    validset_info = get_sepdata_path(os.path.join(data_dir, "valid"))
    testset_info = get_sepdata_path(os.path.join(data_dir, "test"))

    feat = "wm"
    fnum = 1024

    x_train, y_train = load_data(trainset_info, "trainset", feat, fnum)
    x_valid, y_valid = load_data(validset_info, "validset", feat, fnum)
    x_test, y_test = load_data(testset_info, "testset", feat, fnum)

    print(x_train.shape)

    paras = {"booster": "gbtree",
             "objective": "binary:logistic",
             "n_jobs": 6,
             "silent": True,
             "max_depth": 6,
             "learning_rate": 1e-1,
             "n_estimators": 200,
             "reg_lambda": 1e-4,
             "random_state": 72701}

    clf = xgb(**paras)
    clf.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_metric="error", verbose=False)
    y_train_pred = clf.predict(x_train)
    print("Train Accuracy: ", accuracy_score(y_train, y_train_pred))
    y_valid_pred = clf.predict(x_valid)
    print("Valid Accuracy: ", accuracy_score(y_valid, y_valid_pred))
    y_test_pred = clf.predict(x_test)
    print("Test Accuracy: ", accuracy_score(y_test, y_test_pred))

    # plot_importance(clf)
    # pyplot.show()

    importance = clf.feature_importances_
    vip_idx = np.where(importance > 0)[0]
    # print(vip_idx)
    print("Number of important features: ", len(vip_idx))
    vip_x_train = x_train[:, vip_idx]
    vip_x_valid = x_valid[:, vip_idx]
    vip_x_test = x_test[:, vip_idx]
    x_train = vip_x_train
    x_valid = vip_x_valid
    x_test = vip_x_test

    valid_res, test_res = [], []
    ac_res, nc_res = [], []
    ad_idx = np.where(y_test == 1)[0]
    nc_idx = np.where(y_test == 0)[0]
    for i in range(100):
        print("Run ", i + 1)
        model = MLPClassifier(hidden_layer_sizes=(64, ),
                              batch_size=32,
                              max_iter=100, alpha=1e-4,
                              solver="sgd", verbose=0,
                              tol=1e-5,
                              activation="relu",
                              learning_rate_init=1e-3,
                              learning_rate="adaptive")

        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        print("Train Accuracy: ", train_acc)

        y_valid_pred = model.predict(x_valid)
        valid_acc = accuracy_score(y_valid, y_valid_pred)
        print("Valid Accuracy: ", valid_acc)

        y_test_pred = model.predict(x_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        ad_acc = accuracy_score(y_test[ad_idx], y_test_pred[ad_idx])
        nc_acc = accuracy_score(y_test[nc_idx], y_test_pred[nc_idx])
        print("Test Accuracy: ", test_acc)
        print("Test AD Accuracy: ", ad_acc)
        print("Test NC Accuracy: ", nc_acc)

        valid_res.append(valid_acc)
        test_res.append(test_acc)
        ac_res.append(ad_acc)
        nc_res.append(nc_acc)

    print("\nValid ACC: {0:.4f} +/- {1:.4f}".format(np.mean(valid_res), np.std(valid_res)))
    print("Test ACC: {0:.4f} +/- {1:.4f}".format(np.mean(test_res), np.std(test_res)))
    print("Test AD ACC: {0:.4f} +/- {1:.4f}".format(np.mean(ac_res), np.std(ac_res)))
    print("Test NC ACC: {0:.4f} +/- {1:.4f}".format(np.mean(nc_res), np.std(nc_res)))

    test_res = list(test_res)
    acc_set = list(set(test_res))
    acc_set.sort()
    print("Occurrence")
    for acc in acc_set:
        print(acc, "-", test_res.count(acc))
