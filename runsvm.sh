#!/bin/bash
python MiniBatchSVM.py \
    -m svm \
    -t /Share/home/zhangqf6/xukui/data/cryoem/emdb_4A/trainfiles/train_data2.txt \
    -T /Share/home/zhangqf6/xukui/data/cryoem/emdb_4A/trainfiles/test_data2.txt \
    -c 25 \
    --labelstart1 \
    --norm \
    | tee svm_emdb_4A.log
