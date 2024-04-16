#!/usr/bin/env bash

set -e

VERSION=${VERSION:-"2.1"}
DATA_DIR="/data3/wuyini_dataset/AVA_DATASET/annotations"
#
#if [[ ! -d "${DATA_DIR}" ]]; then
#  echo "${DATA_DIR} does not exist. Creating";
#  mkdir -p ${DATA_DIR}
#fi
#
#wget https://s3.amazonaws.com/ava-dataset/annotations/ava_v${VERSION}.zip
unzip -j ava_v${VERSION}.zip -d ${DATA_DIR}/
rm ava_v${VERSION}.zip
