#!/bin/sh

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
