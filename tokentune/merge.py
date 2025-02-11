# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 The Platypus Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied from https://github.com/arielnlee/Platypus/blob/main/merge.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import argparse

"""
This script merges the LoRA/QLoRA weights, obtained via fine-tuning, back into the base model.   
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str,
                        help="name or path of the base model")
    parser.add_argument("--peft_model_path", type=str,
                        help="path to the PEFT model (e.g., LoRA, QLoRA) to merge")
    parser.add_argument("--output_dir", type=str,
                        help="directory to output the merged weights")
    parser.add_argument("--device", type=str, default="auto",
                        help="device")

    return parser.parse_args()


def main():
    args = get_args()

    if args.device == 'auto':
        device_arg = {'device_map': 'auto'}
    else:
        device_arg = {'device_map': {"": args.device}}

    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg
    )

    print(f"Loading PEFT: {args.peft_model_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path, **device_arg)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    model.save_pretrained(f"{args.output_dir}")
    tokenizer.save_pretrained(f"{args.output_dir}")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
