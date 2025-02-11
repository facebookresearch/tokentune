#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/../ || exit  # cd to project root

# NOTE: change the arguments according to your setup
python ./tokentune/merge.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft_model_path ./models/meta-llama-Llama-2-7b-hf-garage-bAInd-Open-Platypus-lora-tokentune-0.3/ \
    --output_dir ./models/meta-llama-Llama-2-7b-hf-garage-bAInd-Open-Platypus-lora-tokentune-0.3/merged
