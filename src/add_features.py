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

        self.dataset = dataset
        self.desc_list = desc_list
        self.volume_type = volume_type
        self.model_dir = best_models_dir
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._resolve_paras()

        self.feat_dir = os.path.join(features_dir, self.out_dir)
        self.create_dir(features_dir, rm=False)
        self.weight_path = os.path.join(best_models_dir, self.weight_name)

        return

    def _resolve_paras(self):
        self.weight_name = self.paras["weight_name"]
        self.model_name = self.paras["model_name"]
        self.scale = self.paras["scale"]
        self.out_dir = self.paras["out_dir"]
        return

    def run(self):
        print("Starting to extract features ...")
        try:
            self._extract()
        except RuntimeError:
            print("Faild to extract features.")
        return

    def _extract(self):

        for data, mode in zip(self.dataset, self.desc_list):
            print("Extract features from ", mode, " set")
            feats_out_dir = os.path.join(self.feat_dir, mode)
            self.create_dir(feats_out_dir, rm=False)

            model = ADDModels(model_name=self.model_name,
                              scale=self.scale).model
            # model.summary()
            model.load_weights(self.weight_path)

            fc1024_dense = Model(inputs=model.input,
                                 outputs=model.get_layer("fts_all").output)
            fc256_dense = Model(inputs=model.input,
                                outputs=model.get_layer("fc2_bn").output)

            for subj in tqdm(data):
                subj_dir, label = subj[0], subj[1]
                for scan in os.listdir(subj_dir):
                    scan_dir = os.path.join(subj_dir, scan)
                    volume_name = [p for p in os.listdir(scan_dir)
                                   if self.volume_type in p][0]
                    volume_path = os.path.join(scan_dir, volume_name)
                    volume_info = volume_path.split("/")

                    label = volume_info[-4]
                    ID = volume_info[-3]
                    idx = volume_info[-2]
                    out_dir = os.path.join(feats_out_dir, label, ID, idx)
                    self.create_dir(out_dir, rm=False)

                    volume = self.load_nii(volume_path)
                    volume = np.expand_dims(volume, axis=0)
                    volume = np.expand_dims(volume, axis=4)

                    fc1024 = fc1024_dense.predict(volume)
                    fc256 = fc256_dense.predict(volume)

                    fc1024_path = os.path.join(out_dir, self.volume_type + "_1024.npy")
                    fc256_path = os.path.join(out_dir, self.volume_type + "_256.npy")

                    np.save(fc1024_path, fc1024)
                    np.save(fc256_path, fc256)

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

    @staticmethod
    def load_nii(path):
        volume = nib.load(path).get_data()
        volume = np.transpose(volume, axes=[2, 0, 1])
        volume = np.rot90(volume, 2)
        obj_idx = np.where(volume > 0)
        volume_obj = volume[obj_idx]
        obj_mean = np.mean(volume_obj)
        obj_std = np.std(volume_obj)
        volume_obj = (volume_obj - obj_mean) / obj_std
        volume[obj_idx] = volume_obj
        return volume


def main(feat_paras_name, volume_type):

    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, pre_paras["data_dir"])
    best_models_dir = os.path.join(parent_dir, pre_paras["best_models_dir"])
    features_dir = os.path.join(parent_dir, pre_paras["features_dir"])

    ad_dir = os.path.join(data_dir, pre_paras["ad_in"])
    nc_dir = os.path.join(data_dir, pre_paras["nc_in"])

    # Load dataset which has been splitted
    data = ADDDataset(ad_dir, nc_dir,
                      subj_sapareted=pre_paras["subj_sapareted"],
                      volume_type=volume_type,
                      pre_trainset_path=pre_paras["pre_trainset_path"],
                      pre_validset_path=pre_paras["pre_validset_path"],
                      pre_testset_path=pre_paras["pre_testset_path"])
    data.run(pre_split=True, only_load_info=True)

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
    help_str = "Select a set of parameters in feat_paras.json."
    parser.add_argument("--paras", action="store", default="whole",
                        dest="feat_paras_name", help=help_str)
    help_str = "Select a volume type in ['whole', 'gm', 'wm', 'csf']."
    parser.add_argument("--volume", action="store", default="whole",
                        dest="volume_type", help=help_str)

    args = parser.parse_args()
    main(args.feat_paras_name, args.volume_type)
