# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import os
import random
import sys
from datetime import datetime

import numpy as np
import peft
import torch
import transformers
from datasets import load_dataset

from llama_tokentune import DataCollatorWithPaddingForPrefix, LlamaPrefixForCausalLM
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

from prompter import Prompter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.file_utils import is_apex_available
from transformers.integrations import TensorBoardCallback
from transformers.utils import logging

transformers.logging.set_verbosity_debug()

logger = logging.get_logger("transformers")
os.environ["WANDB_DISABLED"] = "true"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_config(args):
    logger.info("|{}|".format("=" * 123))
    logger.info("|{:^123}|".format("Configuration"))
    logger.info("|{}|".format("=" * 123))
    for arg in vars(args):
        logger.info("|{:>30} | {:^90}|".format(arg, str(getattr(args, arg))))
    logger.info("|{}|".format("=" * 123))


class GpuMemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

        allocated_memory = torch.cuda.memory_allocated(0) / 1024**2
        reserved_memory = torch.cuda.memory_reserved(0) / 1024**2
        max_reserved_memory = torch.cuda.max_memory_reserved(0) / 1024**2
        if state.is_local_process_zero:
            logger.info(
                "GPU memory: (Allocated) {:.0f}MB, (Reserved) {:.0f}MB (Max) {:.0f}MB".format(
                    allocated_memory, reserved_memory, max_reserved_memory
                )
            )
            torch.cuda.reset_peak_memory_stats()


def main():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument(
        "--global_batch_size",
        default=128,
        type=int,
        help="global batch size. it will be divided in mini-batch for each worker",
    )
    parser.add_argument(
        "--num_workers", default=0, type=int, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=0,
        type=int,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--num_epochs", default=1, type=int, help="number of training epochs"
    )

    # Optimizer & Scheduler
    parser.add_argument(
        "--learning_rate", default=1.5e-4, type=float, help="learning rate calue"
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="applied weight penalization"
    )
    parser.add_argument(
        "--warmup_steps",
        default=3000,
        type=int,
        help="number of warmup accumulation steps",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine",
        type=str,
        help="type of learning rate decay",
    )

    # Efficient Fine-Tuning
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.set_defaults(fp16=False)
    parser.add_argument(
        "--fp16_opt_level",
        default="O1",
        type=str,
        help="level of mixed precision training",
    )
    parser.add_argument(
        "--lora", dest="lora", action="store_true", default=False,
        help="whether to fine-tune using LoRA"
    )
    parser.add_argument(
        "--qlora", dest="qlora", action="store_true", default=False,
        help="whether to fine-tune using QLoRA"
    )
    parser.add_argument(
        "--tokentune", dest="tokentune", action="store_true", default=False,
        help="whether to fine-tune using TokenTune"
    )
    parser.add_argument(
        "--prefix_length",
        default=100,
        type=float,
        help="number or proportion of input tokens to fine-tune",
    )
    parser.add_argument("--bf16", dest="bf16", action="store_true", default=False)
    parser.add_argument(
        "--lora_r", default="8", type=int, help="rank of adaptation layers"
    )
    parser.add_argument(
        "--lora_alpha",
        default="16",
        type=int,
        help="scaling factor of adaptation layers",
    )
    parser.add_argument(
        "--lora_dropout",
        default="0.05",
        type=float,
        help="dropout factor of adaptation layers",
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["gate_proj", "down_proj", "up_proj"],
    )

    # Parallelization
    parser.add_argument("--model_parallel", dest="model_parallel", action="store_true")
    parser.set_defaults(model_parallel=False)

    # Data
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="Instruction dataset on which to fine-tune the model",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        help="path to save models intermediate and final checkpoints",
    )
    parser.add_argument("--logging_dir", type=str, help="path for data logging (tfb)")
    parser.add_argument(
        "--logging_steps",
        default=1000,
        type=int,
        help="number of steps between two logs",
    )
    parser.add_argument(
        "--save_steps",
        default=50000,
        type=int,
        help="number steps between two checkpoints",
    )
    parser.add_argument(
        "--save_total_limit", default=10, type=int, help="maximum number of checkpoints"
    )

    # Model
    parser.add_argument(
        "--model_name_or_path", required=True, type=str, help="model if from hf hub."
    )
    parser.add_argument("--add_eos_token", dest="add_eos_token", action="store_true")
    parser.set_defaults(add_eos_token=False)
    parser.add_argument(
        "--max_length", default=1024, type=int, help="max number of tokens per input"
    )

    parser.add_argument(
        "--prompt_template_name_or_path",
        required=True,
        type=str,
        default="alpaca_template.json",
        help="path to json prompt template",
    )

    args = parser.parse_args()

    log_config(args)  # print config

    set_seed(42)
    fine_tune(args)


def fine_tune(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id = 0
    assert pad != eos, (pad, eos)  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"
    logger.info(f"Setting PAD to {pad}")
    logger.info(f"{args.model_name_or_path} token ids: BOS={bos}, EOS={eos}, PAD={pad}")

    if args.qlora and args.tokentune:
        compute_dtype = (
            torch.float16
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
        )

        setattr(config, "prefix_length", args.prefix_length)
        setattr(config, "use_cache", False)
        model = LlamaPrefixForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            load_in_4bit=True,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=(
                torch.float32
                if args.fp16
                else (torch.bfloat16 if args.bf16 else torch.float32)
            ),
            config=config,
        )
        model.config.torch_dtype = (
            torch.float32
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        )
    elif args.qlora and not args.tokentune:
        logger.info(f"Loading {args.model_name_or_path} with `load_in_4bit=True`.")
        compute_dtype = (
            torch.float16
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            load_in_4bit=True,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=(
                torch.float32
                if args.fp16
                else (torch.bfloat16 if args.bf16 else torch.float32)
            ),
        )
        model.config.torch_dtype = (
            torch.float32
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        )
    elif args.tokentune:  # selective fine-tuning
        logger.info(
            f"Loading LlamaPrefixForCausalLM with `torch_dtype=torch.bfloat16` and `load_in_8bit=False`."
        )
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
        )

        setattr(config, "prefix_length", args.prefix_length)
        setattr(config, "use_cache", False)
        model = LlamaPrefixForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            device_map=0,
            config=config,
        )
    else:
        logger.info(
            f"Loading {args.model_name_or_path} with `torch_dtype=torch.bfloat16` and `load_in_8bit=False`."
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            device_map="auto",
        )

    if args.lora:
        logger.info(f"Applying LoRA layers to {args.lora_target_modules}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        # since we not using gradient checkpointing and 8 bit.
        # hidden states after embeddings do not require grad
        # so we force it to require grad.
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

        if args.qlora:
            # this condition copied from https://github.com/artidoro/qlora/blob/main/qlora.py
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if args.bf16:
                        module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        if args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
        model.print_trainable_parameters()

    if args.model_parallel:  # Distribute model
        logger.info("parallelize model")
        n_gpu = torch.cuda.device_count()
        logger.info("detected {} gpus ".format(n_gpu))
        n_layers_per_device = args.n_layers // n_gpu
        device_map = {
            gpu: list(range(gpu * n_layers_per_device, (gpu + 1) * n_layers_per_device))
            for gpu in range(n_gpu)
        }
        logger.info(device_map)
        model.parallelize(device_map)
    else:
        logger.info("Do not parallelize model")
        torch.cuda.set_device(0)

    device_batch_size = args.global_batch_size // args.gradient_accumulation_steps

    dataset = load_dataset(args.dataset_name_or_path)
    # load data with distributed sampler
    prompter = Prompter(
        tokenizer,
        args.max_length,
        args.prompt_template_name_or_path,
        args.add_eos_token,
    )

    len_before_filetring = len(dataset["train"])
    idx_to_keep = [
        idx
        for idx, s in enumerate(dataset["train"])
        if s["data_source"] not in ["airoboros", "leetcode_ne"]
    ]
    dataset["train"] = dataset["train"].select(indices=idx_to_keep)
    len_after_filetring = len(dataset["train"])
    logger.info(
        f'Filtered {len_before_filetring - len_after_filetring} examples by removing samples from "leetcode_ne" and "airoboros"'
    )

    train_data = dataset["train"].shuffle().map(prompter.tokenize_and_format_input)
    val_data = None

    data_collator_prefix = DataCollatorWithPaddingForPrefix(
        tokenizer,
        prefix_length=args.prefix_length,
        max_length=args.max_length,
    )

    data_dir = (
        args.model_name_or_path.replace("/", "-")
        + "-"
        + args.dataset_name_or_path.replace("/", "-")
    )
    if args.lora:
        data_dir = data_dir + "-" + "lora"
    if args.qlora:
        data_dir = data_dir + "-" + "qlora"
    if args.tokentune:
        prefix_length = (
            str(int(args.prefix_length))
            if args.prefix_length.is_integer()
            else str(args.prefix_length)
        )
        data_dir = data_dir + "-" + "tokentune-" + prefix_length
    model_save_path = os.path.join(args.model_save_path, data_dir)

    print("model:", model)
    # for k, v in model.named_parameters():
    #     if v.requires_grad:
    #         logger.info(f"{k}: {v.shape}")

    training_args = TrainingArguments(
        output_dir=model_save_path,
        per_device_train_batch_size=device_batch_size,
        per_device_eval_batch_size=device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="no",
        eval_steps=None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=1,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator_prefix if args.tokentune else default_data_collator,
        callbacks=[GpuMemoryCallback, TensorBoardCallback],
    )

    # training
    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info("model has {:,} parameters.".format(n_params))
    logger.info(
        "Using trainer parallel mode ... {}".format(training_args.parallel_mode)
    )
    logger.info("Trainer number of cuda devices ... {}".format(training_args.n_gpu))
    start = datetime.now()

    if torch.__version__ >= "2" and sys.platform != "win32":
        logger.info("Compile model with torch 2.0")
        model = torch.compile(model)

    trainer.train()

    logger.info(f"Saving model to {model_save_path}")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    logger.info(">>> Training complete in: " + str(datetime.now() - start))


if __name__ == "__main__":
    logger.info("Using transformers version {}".format(transformers.__version__))
    logger.info("Using peft version {}".format(peft.__version__))
    logger.info("Using torch version {}".format(torch.__version__))
    logger.info("Apex is available ... {}".format(is_apex_available()))

    logger.info("torch cuda is available ... {}".format(torch.cuda.is_available()))
    logger.info("number of cuda devices ... {}".format(torch.cuda.device_count()))

    main()
