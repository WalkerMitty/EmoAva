#!/bin/bash

python3 translate.py \
    -model "xx/output/model.chkpt" \
    -tokenizer_path "bert-base-cased" \
    -save_path "./infer_mid_results" \
    -data_source "xx/EmoAva/dataset" \
    -save_name "result.pt" \
    -max_seq_len 256 \
    -src_len 128 \
    -batch_size 512 \
    -infer_mode "s" \
    -seed 42 \
    -cvae
