# Alzheimer's Disease Detection
# Main script contains whole process.
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
from add_test import ADDTest
from add_train import ADDTrain
from add_dataset import ADDDataset


def main(hyper_paras_name, volume_type):
    '''MAIN

        Main process of Alzheimer's Disease Detection.
        -1- Split dataset for training, validating and testing.
        -2- Train model.
        -3- Test model.

        Inputs:
        -------

        - hyper_paras_name: string, the name of hyperparanters set,
                            which can be found in hyper-paras.json.
        - volume_type: string, one of ["whole", "gm", "wm", "csf"].

    '''

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

    # Set directory to save weights
    weights_save_dir = os.path.join(parent_dir, pre_paras["weights_save_dir"], volume_type)
    # Set directory to save training and validation logs
    logs_save_dir = os.path.join(parent_dir, pre_paras["logs_save_dir"], volume_type)
    # Set directory to save metrics
    results_save_dir = os.path.join(parent_dir, pre_paras["results_save_dir"], volume_type)
    # Set directory where saves pre-trained model
    pre_trained_path = os.path.join(parent_dir, pre_paras["pre_trained_path"])

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
    # Split dataset and save IDs in csv file for each set
    data.run(pre_split=pre_paras["pre_split"],
             save_split=pre_paras["save_split"],
             save_split_dir=pre_paras["save_split_dir"])

    # Training the model
    train = ADDTrain(paras_name=hyper_paras_name,
                     paras_json_path=pre_paras["hyper_paras_json_path"],
                     weights_save_dir=weights_save_dir,
                     logs_save_dir=logs_save_dir,
                     save_best_weights=pre_paras["save_best_weights"],
                     pre_trained_path=pre_trained_path)
    train.run(data)

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
    # python add.py --paras=paras-1 --volume=whole

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
