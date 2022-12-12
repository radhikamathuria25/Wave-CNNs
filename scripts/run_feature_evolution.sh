#!/bin/bash

python3 ../main.py \
    --seed=202211 \
    \
    --num_epoch=1250 \
    --batch_size=32 \
    --lr0=0.01 \
    --lr_a=0.001 \
    --lr_b=0.75 \
    --momentum=0.9 \
    --w_decay=5e-4 \
    --lbl_sm=0.01 \
    \
    --model='resnet50-lrelu' \
    --wavelet='haar' \
    \
    --dataset='fp302u' \
    --num_workers=2 \
    \
    --task='feature_evolve' \
    \
    --datadir='../data/' \
    --logdir='../log/' \
    --ptdir='../pretrain/' \
    --log_filename='train' \
    --init_weights=0 \
    --resume_train=0 \
    \
    --exp_label='waveeff' \
    \
    --gpu=3
