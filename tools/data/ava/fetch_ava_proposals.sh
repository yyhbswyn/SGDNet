#!/usr/bin/env bash

set -e

DATA_DIR="/data3/wuyini_dataset/AVA_DATASET/annotations"

wget https://download.openmmlab.com/mmaction/dataset/ava/ava_dense_proposals_train.FAIR.recall_93.9.pkl -P ${DATA_DIR}
wget https://download.openmmlab.com/mmaction/dataset/ava/ava_dense_proposals_val.FAIR.recall_93.9.pkl -P ${DATA_DIR}
wget https://download.openmmlab.com/mmaction/dataset/ava/ava_dense_proposals_test.FAIR.recall_93.9.pkl -P ${DATA_DIR}
