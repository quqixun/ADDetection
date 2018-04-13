from __future__ import print_function


import os
import numpy as np
import pandas as pd
import nibabel as nib
from random import seed, shuffle
from keras.utils import to_categorical


class ADDDataset(object):

    def __init__(self,
                 ad_dir, nc_dir,
                 volume_type="whole",
                 train_prop=0.7,
                 valid_prop=0.15,
                 random_state=0,
                 is_augment=False,
                 pre_trainset_path=None,
                 pre_validset_path=None,
                 pre_testset_path=None,
                 data_format=".nii.gz"):
        '''__INIT__
        '''

        self.ad_dir = ad_dir
        self.nc_dir = nc_dir
        self.volume_type = volume_type

        self.train_prop = train_prop
        self.valid_prop = valid_prop
        self.random_state = int(random_state)
        self.is_augment = is_augment

        self.pre_trainset = pre_trainset_path
        self.pre_validset = pre_validset_path
        self.pre_testset = pre_testset_path
        self.data_format = data_format

        self.trainset = None
        self.validset = None
        self.testset = None

        self.train_x, self.train_y = None, None
        self.valid_x, self.valid_y = None, None
        self.test_x, self.test_y = None, None

        return

    def run(self, pre_split=False,
            save_split=False,
            save_split_dir=None,
            only_load_info=False):
        print("\nSplitting dataset to train, valide and test.\n")
        self.trainset, self.validset, self.testset = \
            self._get_pre_datasplit() if pre_split else \
            self._get_new_datasplit()

        if only_load_info:
            return

        self._load_dataset(self.trainset, self.validset,
                           self.testset, self.volume_type)

        if save_split and (not pre_split):
            self.save_split_dir = save_split_dir
            self._save_dataset(self.trainset, self.validset, self.testset)
        return

    def _get_pre_datasplit(self):
        paras = {"ad_dir": self.ad_dir,
                 "nc_dir": self.nc_dir,
                 "data_format": self.data_format,
                 "csv_path": None}

        paras["csv_path"] = self.pre_trainset
        trainset = self.load_datasplit(**paras)

        paras["csv_path"] = self.pre_validset
        validset = self.load_datasplit(**paras)

        paras["csv_path"] = self.pre_testset
        testset = self.load_datasplit(**paras)

        return trainset, validset, testset

    def _get_new_datasplit(self):
        paras = {"label": None,
                 "dir_path": None,
                 "random_state": self.random_state}

        paras["label"], paras["dir_path"] = 1, self.ad_dir
        ad_subjects = self.get_subjects_path(**paras)

        paras["label"], paras["dir_path"] = 0, self.nc_dir
        nc_subjects = self.get_subjects_path(**paras)

        paras = {"subjects": None,
                 "train_prop": self.train_prop,
                 "valid_prop": self.valid_prop}

        paras["subjects"] = ad_subjects
        ad_train, ad_valid, ad_test = self.split_dataset(**paras)

        paras["subjects"] = nc_subjects
        nc_train, nc_valid, nc_test = self.split_dataset(**paras)

        trainset = ad_train + nc_train
        validset = ad_valid + nc_valid
        testset = ad_test + nc_test

        return trainset, validset, testset

    def _load_dataset(self, trainset, validset, testset, volume_type):

        self.test_x, test_y = self.load_data(testset, "test set", volume_type)
        self.test_y = to_categorical(test_y, num_classes=2)

        self.valid_x, valid_y = self.load_data(validset, "valid set", volume_type)
        self.valid_y = to_categorical(valid_y, num_classes=2)

        train_x, train_y = self.load_data(trainset, "train set", volume_type)
        if self.is_augment:
            train_x, train_y = self.augment(train_x, train_y)
        self.train_x = train_x
        self.train_y = to_categorical(train_y, num_classes=2)

        return

    def _save_dataset(self, trainset, validset, testset):
        ap = str(self.random_state) + ".csv"
        trainset_path = os.path.join(self.save_split_dir, "trainset_" + ap)
        validset_path = os.path.join(self.save_split_dir, "validset_" + ap)
        testset_path = os.path.join(self.save_split_dir, "testset_" + ap)

        self.save_datasplit(trainset, trainset_path)
        self.save_datasplit(validset, validset_path)
        self.save_datasplit(testset, testset_path)

        return

    @staticmethod
    def load_datasplit(ad_dir, nc_dir, csv_path,
                       data_format=".nii"):
        '''LOAD_DATASPLIT
        '''
        df = pd.read_csv(csv_path)
        IDs = df["ID"].values.tolist()
        labels = df["label"].values.tolist()
        info = []
        for ID, label in zip(IDs, labels):
            target_dir = ad_dir if label else nc_dir
            subj_dir = os.path.join(target_dir, ID)
            info.append([subj_dir, label])
        return info

    @staticmethod
    def save_datasplit(dataset, to_path):
        IDs, labels = [], []
        for i in dataset:
            IDs.append(i[0].split("/")[-1].split(".")[0])
            labels.append(i[1])

        df = pd.DataFrame(data={"ID": IDs, "label": labels})
        df.to_csv(to_path, index=False)
        return

    @staticmethod
    def get_subjects_path(dir_path, label, random_state=0):
        subjects = os.listdir(dir_path)
        seed(random_state)
        shuffle(subjects)
        subjects_paths = []
        for subject in subjects:
            subject_path = os.path.join(dir_path, subject)
            subjects_paths.append([subject_path, label])
        return subjects_paths

    @staticmethod
    def split_dataset(subjects, train_prop=0.7, valid_prop=0.15):
        subj_num = len(subjects)
        train_valid_num = subj_num * (train_prop + valid_prop)
        train_valid_idx = int(round(train_valid_num))
        testset = subjects[train_valid_idx:]

        valid_idx = int(round(subj_num * valid_prop))
        validset = subjects[:valid_idx]
        trainset = subjects[valid_idx:train_valid_idx]
        return trainset, validset, testset

    @staticmethod
    def load_data(dataset, mode, volume_type):

        def load_nii(nii_path):
            volume = nib.load(nii_path).get_data()
            volume = np.transpose(volume, axes=[2, 0, 1])
            volume = np.rot90(volume, 2)

            obj_idx = np.where(volume > 0)
            volume_obj = volume[obj_idx]
            obj_mean = np.mean(volume_obj)
            obj_std = np.std(volume_obj)
            volume_obj = (volume_obj - obj_mean) / obj_std
            volume[obj_idx] = volume_obj
            return volume

        x, y = [], []
        print("Loading {} data ...".format(mode))
        for subject in dataset:
            subj_path, label = subject[0], subject[1]
            if os.path.isdir(subj_path):
                for scan in os.listdir(subj_path):
                    scan_dir = os.path.join(subj_path, scan)
                    volume_name = [p for p in os.listdir(scan_dir)
                                   if volume_type in p][0]
                    volume_path = os.path.join(scan_dir, volume_name)
                    volume = load_nii(volume_path)
                    volume = np.expand_dims(volume, axis=3)
                    x.append(volume.astype(np.float32))
                    y.append(label)
            else:
                volume = load_nii(subj_path)
                volume = np.expand_dims(volume, axis=3)
                x.append(volume.astype(np.float32))
                y.append(label)

        x = np.array(x)
        y = np.array(y).reshape((-1, 1))

        return x, y

    @staticmethod
    def augment(train_x, train_y):
        print("Do Augmentation on nc Samples ...")
        train_x_aug, train_y_aug = [], []
        for i in range(len(train_y)):
            train_x_aug.append(train_x[i])
            train_y_aug.append(train_y[i])
            train_x_aug.append(np.fliplr(train_x[i]))
            train_y_aug.append(train_y[i])
        train_x = np.array(train_x_aug)
        train_y = np.array(train_y_aug).reshape((-1, 1))

        return train_x, train_y


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data", "adni_subj")
    ad_dir = os.path.join(data_dir, "AD")
    nc_dir = os.path.join(data_dir, "NC")

    # Subject-saparated
    # Load and split dataset
    data = ADDDataset(ad_dir, nc_dir,
                      volume_type="whole",
                      train_prop=0.7,
                      valid_prop=0.15,
                      random_state=1)
    data.run(save_split=False)
    print(data.train_x.shape, data.train_y.shape)

    # Load dataset which has been splitted
    data = ADDDataset(ad_dir, nc_dir,
                      volume_type="whole",
                      pre_trainset_path="DataSplit/trainset.csv",
                      pre_validset_path="DataSplit/validset.csv",
                      pre_testset_path="DataSplit/testset.csv")
    data.run(pre_split=True)
    print(data.train_x.shape, data.train_y.shape)

    data_dir = os.path.join(parent_dir, "data", "adni")
    ad_dir = os.path.join(data_dir, "AD")
    nc_dir = os.path.join(data_dir, "NC")

    # None subject-saparated
    # Load and split dataset
    data = ADDDataset(ad_dir, nc_dir,
                      volume_type="whole",
                      train_prop=0.7,
                      valid_prop=0.15,
                      random_state=0)
    data.run(save_split=False)
    print(data.train_x.shape, data.train_y.shape)
