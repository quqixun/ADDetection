# Alzheimer's Disease Detection
# Extracted features from brain image
# using trained model.
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


from __future__ import division
from __future__ import print_function


import os
import json
import argparse
import numpy as np
import nibabel as nib

from tqdm import *
from keras.models import Model
from add_models import ADDModels
from add_dataset import ADDDataset


class ADDFeatures(object):

    def __init__(self, dataset, desc_list, volume_type,
                 features_dir, best_models_dir,
                 paras_name, paras_json_path):
        '''__INIT__

            Set parameters before extracting features.

            Inputs:
            -------

            - dataset: list, contains partition information of
                       training set, validation set and testing set.
            - desc_list: list of strings, ["train", "valid", "test"].
            - volume_type: string, indicates which volume used
                           to extract features.
            - features_dir: string, path of output directory.
            - best_models_dir: string, path of directory of trained models,
                               they will be used to extract features.
            - paras_name: string, name of parameters set,
                          can be found in feat_paras.json.
            - paras_json_path: string, path of file which provides
                               paramters, "feat_paras.json" in this project.

        '''

        self.dataset = dataset
        self.desc_list = desc_list
        self.volume_type = volume_type
        self.model_dir = best_models_dir

        # Load parameters from feat_paras.json
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._load_paras()

        # Creat directory to save features
        self.feat_dir = os.path.join(features_dir, self.out_dir)
        self.create_dir(features_dir, rm=False)

        # Generate path of trained model
        self.weight_path = os.path.join(best_models_dir, self.weight_name)

        return

    def _load_paras(self):
        '''_LOAD_PARAS

            Load parameters from feat_paras.json.

        '''

        self.weight_name = self.paras["weight_name"]  # weights file's name
        self.model_name = self.paras["model_name"]    # model's structure
        self.scale = self.paras["scale"]              # use which scale
        self.out_dir = self.paras["out_dir"]          # output directory

        return

    def run(self):
        '''RUN

            Main function to extract features.

        '''

        print("Starting to extract features ...")
        try:
            self._extract()
        except RuntimeError:
            print("Faild to extract features.")

        return

    def _extract(self):
        '''_EXTRACT

            Extract features from each scan and
            save features as npy files.

            Outputs:
            --------

            - [self.volume_type]_1024.npy, the fusion of
              four scales features, its length is 1024.

            - [self.volume_type]_256.npy, the final densely layer
              befor output layer, its length is 256.

        '''

        for data, mode in zip(self.dataset, self.desc_list):
            # For each partition
            print("Extract features from ", mode, " set")

            # Create output directory to save features
            feats_out_dir = os.path.join(self.feat_dir, mode)
            self.create_dir(feats_out_dir, rm=False)

            # Build model and load weights
            model = ADDModels(model_name=self.model_name,
                              scale=self.scale).model
            model.load_weights(self.weight_path)

            # Set output from model
            fc1024_dense = Model(inputs=model.input,
                                 outputs=model.get_layer("fts_all").output)
            fc256_dense = Model(inputs=model.input,
                                outputs=model.get_layer("fc2_bn").output)

            for subj in tqdm(data):
                # For each subject in one partition
                subj_dir, label = subj[0], subj[1]
                for scan in os.listdir(subj_dir):
                    # For each scan of one subject
                    scan_dir = os.path.join(subj_dir, scan)

                    # Volume file name which contains self.volume_type
                    volume_name = [p for p in os.listdir(scan_dir)
                                   if self.volume_type in p][0]
                    # Full path of volume
                    volume_path = os.path.join(scan_dir, volume_name)

                    # Split volume path to extract information, including
                    # label, subject ID, scan index
                    # Use these information to generate output folder path
                    volume_info = volume_path.split("/")
                    label = volume_info[-4]
                    ID = volume_info[-3]
                    idx = volume_info[-2]
                    out_dir = os.path.join(feats_out_dir, label, ID, idx)
                    self.create_dir(out_dir, rm=False)

                    # Load one image in shape [112, 96, 96]
                    # Expand its dimension to [1, 112, 96, 96, 1]
                    volume = self.load_nii(volume_path)
                    volume = np.expand_dims(volume, axis=0)
                    volume = np.expand_dims(volume, axis=4)

                    # Extract featrures
                    fc1024 = fc1024_dense.predict(volume)
                    fc256 = fc256_dense.predict(volume)

                    # Save features into npy files
                    fc1024_path = os.path.join(out_dir, self.volume_type + "_1024.npy")
                    fc256_path = os.path.join(out_dir, self.volume_type + "_256.npy")
                    np.save(fc1024_path, fc1024)
                    np.save(fc256_path, fc256)

        return

    @staticmethod
    def load_paras(paras_json_path, paras_name):
        '''LOAD_PARAS

            Load parameters from json file.
            See feat_paras.json.

            Inputs:
            -------

            - paras_name: string, name of parameters set,
                          can be found in feat_paras.json.
            - paras_json_path: string, path of file which provides paramters,
                               "feat_paras.json" in this project.

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

    @staticmethod
    def load_nii(path):
        '''LOAD_NII

            Load images from given path.

            Input:
            ------

            - path: string, path of image to be loaded.

            Output:
            -------

            - volume: numpy ndarray, loaded image.

        '''

        # Load images and rotate to standard space
        volume = nib.load(path).get_data()
        volume = np.transpose(volume, axes=[2, 0, 1])
        volume = np.rot90(volume, 2)

        # Extract mean and std from brain object
        obj_idx = np.where(volume > 0)
        volume_obj = volume[obj_idx]
        obj_mean = np.mean(volume_obj)
        obj_std = np.std(volume_obj)
        # Normalize brain object
        volume_obj = (volume_obj - obj_mean) / obj_std
        volume[obj_idx] = volume_obj

        return volume


def main(feat_paras_name, volume_type):
    '''MAIN

        Main process to textract features.

        Inputs:
        -------

        - feat_paras_name: string, the name of parameters set,
                            which can be found in feat_paras.json.
        - volume_type: string, one of ["whole", "gm", "wm", "csf"].

    '''

    # Basic configuration sin pre_paras.json
    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    # Set directory paths of trained models
    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, pre_paras["data_dir"])
    best_models_dir = os.path.join(parent_dir, pre_paras["best_models_dir"])
    features_dir = os.path.join(parent_dir, pre_paras["features_dir"])

    # Set directory of input images
    ad_dir = os.path.join(data_dir, pre_paras["ad_in"])
    nc_dir = os.path.join(data_dir, pre_paras["nc_in"])

    # Load dataset which has been splitted
    data = ADDDataset(ad_dir, nc_dir,
                      subj_separated=pre_paras["subj_separated"],
                      volume_type=volume_type,
                      pre_trainset_path=pre_paras["pre_trainset_path"],
                      pre_validset_path=pre_paras["pre_validset_path"],
                      pre_testset_path=pre_paras["pre_testset_path"])
    data.run(pre_split=True, only_load_info=True)

    # List of pertition information and dexcription
    dataset = [data.trainset, data.validset, data.testset]
    desc_list = ["train", "valid", "test"]

    feat = ADDFeatures(dataset, desc_list,
                       volume_type=volume_type,
                       features_dir=features_dir,
                       best_models_dir=best_models_dir,
                       paras_name=feat_paras_name,
                       paras_json_path=pre_paras["feat_paras_json_path"])
    feat.run()

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Set name of paramters set in feat_paras.json
    help_str = "Select a set of parameters in feat_paras.json."
    parser.add_argument("--paras", action="store", default="whole",
                        dest="feat_paras_name", help=help_str)

    # Select one tissue to extract feartures
    help_str = "Select a volume type in ['whole', 'gm', 'wm', 'csf']."
    parser.add_argument("--volume", action="store", default="whole",
                        dest="volume_type", help=help_str)

    args = parser.parse_args()
    main(args.feat_paras_name, args.volume_type)
