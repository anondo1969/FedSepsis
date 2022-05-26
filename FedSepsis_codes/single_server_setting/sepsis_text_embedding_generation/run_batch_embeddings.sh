#!/bin/sh

CUDA_VISIBLE_DEVICES=3 python3 batch_embeddings.py emily &>> emily_log.txt &

CUDA_VISIBLE_DEVICES=3 python3 batch_embeddings.py huang &>> huang_log.txt &


