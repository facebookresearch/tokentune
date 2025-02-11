#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/../../ || exit  # cd to project root

python -u ./tokentune/finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name_or_path garage-bAInd/Open-Platypus \
    --prompt_template_name_or_path ./tokentune/alpaca_template.json \
    --global_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --num_epochs 1 \
    --num_workers 4 \
    --learning_rate 4e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --model_save_path ./models/ \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_length 2048 \
    --tokentune \
    --prefix_length 0.3 \
    "$@"
