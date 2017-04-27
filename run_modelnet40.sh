#!/bin/bash
datadir="/data/kuixu/data"
da="modelnet40_60x"
python MiniBatchSVM.py \
    -m svm \
    -t $datadir/${da}/train.list \
    -T $datadir/${da}/test.list \
    -b 256 \
    -c 40 \
    #| tee svm_${da}.log
    #--labelstart1 \
    #--norm \
