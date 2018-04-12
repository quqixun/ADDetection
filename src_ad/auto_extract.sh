#!/bin/sh

# Whole
python features.py --stream whole --data train
python features.py --stream whole --data valid
python features.py --stream whole --data test

# GM
python features.py --stream gm --data train
python features.py --stream gm --data valid
python features.py --stream gm --data test

# WM
python features.py --stream wm --data train
python features.py --stream wm --data valid
python features.py --stream wm --data test

# CSF
python features.py --stream csf --data train
python features.py --stream csf --data valid
python features.py --stream csf --data test
