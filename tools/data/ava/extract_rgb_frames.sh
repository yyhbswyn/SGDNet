#!/usr/bin/env bash

cd ../
python build_rawframes.py /data3/wuyini_dataset/AVA_DATASET/videos/train_15min/ /data3/wuyini_dataset/AVA_DATASET/train_rawframes/ --task rgb --level 1 --mixed-ext
echo "Genearte raw frames (RGB only)"

cd ava/
