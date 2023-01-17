#!/bin/sh

#author: Mahbub Ul Alam (mahbub@dsv.su.se)
#version: 1.0+
#copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
#license : MIT License


CUDA_VISIBLE_DEVICES=3 python3 batch_embeddings.py emily &>> emily_log.txt &

CUDA_VISIBLE_DEVICES=3 python3 batch_embeddings.py huang &>> huang_log.txt &


