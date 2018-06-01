#!/bin/sh


# Alzheimer's Disease Detection
# Commands for extracting features.
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

# Commands in this script are used to extract features
# from different tissues by different models.
# The formation of command is:
# python add_features.py --paras=paras_name --volume=volume_type
# Parameters:
# - paras: set of parameters in feat_paras.json, which indicates
#          the name of trained weights and the directory for outputs.
#          One of following choices:
#          ---------------------------------------------------------
#           Model Name    |    Volume Type    |   Pre-Trained Model
#          ---------------------------------------------------------
#              pre        |       whole       |         None
#              gpre       |        GM         |         pre
#              wpre       |        WM         |         pre
#              cpre       |        CSF        |         pre
#              gnew       |        GM         |         None
#              wnew       |        WM         |         None
#              cnew       |        CSF        |         None
#          ---------------------------------------------------------
#           Model Name    |                Comment
#          ---------------------------------------------------------
#              gbest      |        best one of [gpre, gnew]
#              wbest      |        best one of [wpre, wnew]
#              cbest      |        best one of [cpre, cnew]
#          ---------------------------------------------------------
# - volume: one type volume of "whole", "gm", "wm" or "csf"


python add_features.py --paras=pre --volume=whole
python add_features.py --paras=pre --volume=gm
python add_features.py --paras=pre --volume=wm
python add_features.py --paras=pre --volume=csf

python add_features.py --paras=gpre --volume=gm
python add_features.py --paras=wpre --volume=wm
python add_features.py --paras=cpre --volume=csf

python add_features.py --paras=gnew --volume=gm
python add_features.py --paras=wnew --volume=wm
python add_features.py --paras=cnew --volume=csf

python add_features.py --paras=gbest --volume=gm
python add_features.py --paras=wbest --volume=wm
python add_features.py --paras=cbest --volume=csf
