from __future__ import print_function

import os
import json
import shutil
import argparse
from tqdm import *
import numpy as np
import nibabel as nib
from models import *
from random import seed, shuffle


TRAIN_VALID_PROP = 0.85
VALID_PROP = 0.15
TEST_PROP = 0.15
VOLUME_SIZE = [1, 112, 96, 96, 1]


def get_data_path(dir_path, volume_type, label, SEED):
    subjects = os.listdir(dir_path)
    seed(SEED)
    shuffle(subjects)
    subj_scan_paths = []
    for subject in subjects:
        subj_dir = os.path.join(dir_path, subject)
        for scan in os.listdir(subj_dir):
            scan_path = os.path.join(subj_dir, scan, volume_type + ".nii.gz")
            subj_scan_paths.append([scan_path, label])
    return subj_scan_paths


def get_dataset(subjects, valid=False):
    subj_num = len(subjects)
    train_idx = int(round(subj_num * TRAIN_VALID_PROP))

    test_set = subjects[train_idx:]
    if valid:
        valid_idx = int(round(subj_num * VALID_PROP))
        valid_set = subjects[:valid_idx]
        train_set = subjects[valid_idx:train_idx]
        return train_set, valid_set, test_set
    else:
        train_set = subjects[:train_idx]
        return train_set, test_set


def get_sepdata_path(data_dir, volume_type):
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
                scan_path = os.path.join(scan_dir, volume_type + ".nii.gz")
                data_info.append([scan_path, label])

    shuffle(data_info)
    return data_info


def create_dir(path, rm=True):
    if os.path.isdir(path):
        if rm:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
    return


def load_nii(path, wmean=None, wstd=None):
    volume = nib.load(path).get_data()
    volume = np.transpose(volume, axes=[2, 0, 1])
    volume = np.rot90(volume, 2)
    obj_idx = np.where(volume > 0)
    obj = volume[obj_idx]
    if (wmean is None) or (wstd is None):
        wmean, wstd = np.mean(obj), np.std(obj)
    obj = (obj - wmean) / wstd
    volume[obj_idx] = obj
    return volume


def extract_features(info, volume_type, weight_path,
                     feature_dir, model_type, mode):
    print("Extract features from ", mode, " set")
    feats_dir = os.path.join(feature_dir, mode)
    create_dir(feats_dir, rm=False)

    if model_type == "pyramid4":
        model = pyramid4(pool="max")

    model.summary()
    model.load_weights(weight_path)

    fc1024_dense = Model(inputs=model.input,
                         outputs=model.get_layer("concatenate_1").output)
    fc256_dense = Model(inputs=model.input,
                        outputs=model.get_layer("batch_normalization_16").output)

    for subj in tqdm(info):
        volume_path = subj[0]
        volume_info = volume_path.split("/")
        # print(volume_info)
        label_dir = volume_info[-4]
        id_dir = volume_info[-3]
        index_dir = volume_info[-2]
        out_dir = os.path.join(feats_dir, label_dir, id_dir, index_dir)
        create_dir(out_dir, rm=False)

        # whole_path = "/".join(volume_info[:-1] + ["whole.nii.gz"])
        # whole = nib.load(whole_path).get_data()
        # obj_idx = np.where(whole > 0)
        # obj = whole[obj_idx]
        # wmean, wstd = np.mean(obj), np.std(obj)
        # volume = load_nii(volume_path, wmean, wstd)

        volume = load_nii(volume_path)

        volume = np.expand_dims(volume, axis=0)
        volume = np.expand_dims(volume, axis=4)

        fc1024 = fc1024_dense.predict(volume)
        fc256 = fc256_dense.predict(volume)

        fc1024_path = os.path.join(out_dir, volume_type + "_1024.npy")
        fc256_path = os.path.join(out_dir, volume_type + "_256.npy")

        np.save(fc1024_path, fc1024)
        np.save(fc256_path, fc256)

    return


def load_paras(file_path, stream):
    paras = json.load(open(file_path))
    return paras[stream]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    stream_help_str = "Select a stream, gm, wm or csf."
    parser.add_argument("--stream", action="store", default="gm",
                        dest="stream", help=stream_help_str)

    data_help_str = "Select a dataset, tain, valid or test."
    parser.add_argument("--data", action="store", default="train",
                        dest="data", help=data_help_str)

    args = parser.parse_args()
    stream = args.stream
    dataset = args.data

    paras_path = os.path.join(os.getcwd(), "features.json")
    paras = load_paras(paras_path, stream)
    weight_name = paras["weight_name"]
    volume_type = paras["volume_type"]
    data_type = paras["data_type"]
    model_type = paras["model_type"]

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data", data_type)
    if data_type == "adni_pve_sep":
        dataset_dir = os.path.join(data_dir, dataset)
        dataset_info = get_sepdata_path(dataset_dir, volume_type)
    else:
        ad_dir = os.path.join(data_dir, "AD")
        nc_dir = os.path.join(data_dir, "NC")

        SEED = paras["seed"]
        ad_subjects = get_data_path(ad_dir, volume_type, 1, int(SEED))
        nc_subjects = get_data_path(nc_dir, volume_type, 0, int(SEED))

        ad_train, ad_valid, ad_test = get_dataset(ad_subjects, True)
        nc_train, nc_valid, nc_test = get_dataset(nc_subjects, True)

        if dataset == "train":
            dataset_info = ad_train + nc_train
        elif dataset == "valid":
            dataset_info = ad_valid + nc_valid
        elif dataset == "test":
            dataset_info = ad_test + nc_test
        else:
            print("Wrong dataset!")
            raise SystemExit

    weight_path = os.path.join(parent_dir, "best_models", weight_name)
    feature_dir = os.path.join(parent_dir, "features", "pre")

    extract_features(dataset_info, volume_type,
                     weight_path, feature_dir, model_type, dataset)
