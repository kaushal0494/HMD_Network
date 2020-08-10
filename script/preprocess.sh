#!/usr/bin/env bash

python3 -u ../preprocess.py \
        -train_dir=../data/feature_data/race_train_large_features.json \
        -valid_dir=../data/feature_data/race_dev_large_features.json \
        -save_data=../data/processed \
        -share_vocab \
        -total_token_length=550 \
        -src_seq_length=60 \
        -src_sent_length=40 \
        -lower \
        #-feat_name=pos_ner_dep_lemma \
        #-n_feats=4 \
        #-src_vocab_size=200 \
        #-log_file=../logs/preprocess.log

python3 ../embeddings_to_torch.py \
       -emb_file_enc=../../../embeddings/glove.840B.300d.txt \
       -emb_file_dec=../../../embeddings/glove.840B.300d.txt \
       -output_file=../data/processed.glove \
       -dict_file=../data/processed.vocab.pt
        
