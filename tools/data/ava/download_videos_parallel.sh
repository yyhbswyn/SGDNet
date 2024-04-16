#!/usr/bin/env bash

set -e

DATA_DIR="/data3/wuyini_dataset/AVA_DATASET/videos"
ANNO_DIR="/data3/wuyini_dataset/AVA_DATASET/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt -P ${ANNO_DIR}

python download_videos_parallel.py ${ANNO_DIR}/ava_file_names_trainval_v2.1.txt ${DATA_DIR}
