#!/bin/sh


# Alzheimer's Disease Detection
# Commands for refination by fusing
# and selecting features of different
# types of tissues.
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


# Refination
# Command:
# python add_refine.py --paras=paras_name
# Parameters:
# - paras: parameters set in rfn_paras.json

python add_refine.py --paras=refine-1
