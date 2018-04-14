#!/bin/sh

python add.py --paras=paras-1 --volume=whole
python add.py --paras=paras-1 --volume=gm
python add.py --paras=paras-1 --volume=wm
python add.py --paras=paras-1 --volume=csf

python add.py --paras=paras-2 --volume=gm
python add.py --paras=paras-2 --volume=wm
python add.py --paras=paras-2 --volume=csf
