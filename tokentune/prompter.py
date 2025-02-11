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
import json
from typing import Union


class Prompter(object):
    """Adapted from https://github.com/arielnlee/Platypus/blob/main/finetune.py"""

    def __init__(self, tokenizer, max_length, template_name_or_path: str = "alpaca_template.json", add_eos_token=True):
        with open(template_name_or_path) as f:
            self.template = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos_token = add_eos_token

    def prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

    def tokenize(self, text):
        result = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,  # "max_length"
            return_tensors=None,
            truncation=True,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_length
            and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def tokenize_and_format_input(self, input_sample):
        full_prompt = self.prompt(
            input_sample["instruction"],
            input_sample["input"],
            input_sample["output"])
        
        tokenized_full_prompt = self.tokenize(full_prompt)
        
        user_prompt = self.prompt(
            input_sample["instruction"], input_sample["input"])
        
        tokenized_user_prompt = self.tokenize(user_prompt)
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if self.add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
        return tokenized_full_prompt
