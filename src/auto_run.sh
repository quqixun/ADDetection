#!/bin/sh


# Alzheimer's Disease Detection
# Commands for training and testing models.
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

#
# Section 1
#
# Train and test model using subject-separated samples
# Command:
# python add.py --paras=paras_name --volume=volume_type
# Parameters:
# - paras: hyperparameters set in hyper_paras.json
# - volume: one type volume of "whole", "gm", "wm" or "csf"

# Train model from scratch
# python add.py --paras=paras-1 --volume=whole
# python add.py --paras=paras-1 --volume=gm
# python add.py --paras=paras-1 --volume=wm
# python add.py --paras=paras-1 --volume=csf

# Train model based on pre-trained weights
python add.py --paras=paras-2 --volume=gm
python add.py --paras=paras-2 --volume=wm
python add.py --paras=paras-2 --volume=csf


# To train the model using non-subject-separated samples,
# in pre_paras.json:
# - change "data_dir" to the directory which
#   contains non-separated samples, in this case,
#   is "data/adni".
#   NOTE: not the absolute path, but the relative
#         path within project directory
# - change "subj_separated" to False

# python add.py --paras=paras-3 --volume=whole


#
# Section 2
#
# Train and test model respectively
# Commands:
# python add_train.py --paras=paras_name --volume=volume_type
# python add_test.py --paras=paras_name --volume=volume_type
# Same parameters as in Section 1

# python add_train.py --paras=paras-1 --volume=whole
# python add_test.py --paras=paras-1 --volume=whole
