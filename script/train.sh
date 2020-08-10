#!/bin/bash

#PROJ=/home/evanyfgao/Distractor-Generation-RACE

export CUDA_VISIBLE_DEVICES=5
MODEL=$1
DATE=$2

python3 -u ../train_single.py \
        -word_vec_size=300 \
        -share_embeddings \
        -rnn_size=700 \
        -word_encoder_layers=2 \
        -sent_encoder_layers=1 \
        -question_init_layers=2 \
        -dec_layers=2 \
        -lambda_question=0.5 \
        -lambda_answer=-1 \
        -data=../data/processed \
        -save_model=../data/model/${DATE}_${MODEL} \
        -save_checkpoint_steps=3000 \
        -gpuid=0 \
        -pre_word_vecs_enc=../data/processed.glove.enc.pt \
        -pre_word_vecs_dec=../data/processed.glove.dec.pt \
        -batch_size=8 \
        -valid_steps=3000 \
        -valid_batch_size=8 \
        -train_steps=250000  \
        -optim=adagrad \
        -adagrad_accumulator_init=0.1 \
        -learning_rate=0.1 \
        -learning_rate_decay=0.5 \
        -start_decay_steps=150000 \
        -decay_steps=10000 \
        -seed=1995 \
        -report_every=600 \
        #-train_from=../data/model/__step_30.pt \
        #-feat_vec_size=20 \
        #-tensorboard \
        #-tensorboard_log_dir ../data/runs  \
        #-log_file=../logs/train_single.log \

