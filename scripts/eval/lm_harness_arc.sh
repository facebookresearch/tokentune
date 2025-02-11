#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/../../ || exit  # cd to project root

# NOTE: replace this variable with the model to evaluate
MODEL="meta-llama-Llama-2-7b-hf-garage-bAInd-Open-Platypus-lora-tokentune-0.3/merged"
MODEL_ARGS="pretrained=./models/${MODEL}"
#MODEL_ARGS="pretrained=./models/${MODEL},dtype=float16"  # used for experiments in the paper; results mostly the same

accelerate launch -m lm_eval \
  --model hf-auto \
  --model_args "${MODEL_ARGS}" \
  --tasks arc_challenge \
  --batch_size 20 \
  --output_path ./results/"${MODEL}"/arc_challenge_25shot.json \
  --device cuda \
  --num_fewshot 25
