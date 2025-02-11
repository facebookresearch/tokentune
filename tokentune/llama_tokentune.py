# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 The Hugging Face Team
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
import math
from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaConfig,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging, ModelOutput, PaddingStrategy


logger = logging.get_logger(__name__)


"""
In this file, we apply the proposed TokenTune method to the Llama model implementation from the Transformers library.
For each input sequence, TokenTune randomly selects k tokens, and organizes the input into two groups, 
one containing theselected k tokens, and the other containing the remaining unselected tokens. 
Note that this reordering does not impact the computation since positional information is encoded in the hidden states.
Specifically, the "DataCollatorWithPaddingForPrefix" class handles the selection and reordering of input tokens.  
Below, code related to selected tokens is marked with "prefix", indicating that these tokens appear at the beginning of 
the sequence after reordering. For example, "hidden_states_prefix" stores the hidden states of the selected tokens, 
while "hidden_states" stores those of the other unselected tokens. 
The rest of the code, which does not require separate processing of selected and unselected tokens, 
follows the same operations as the original Transformer implementation.
For details on how TokenTune performs the forward pass for selected and unselected tokens in attention layers, 
as well as dense and normalization layers, refer to Section 3 of the paper, particularly Equations (6)–(11) and Algorithm 1.
"""


@dataclass
class DataCollatorWithPaddingForPrefix:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    prefix_length: Union[int, float] = 16

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_batch = []
        position_ids_all = []
        position_ids_prefix_all = []

        for i in range(len(features)):
            input_ids = []
            attention_mask = []
            position_ids = []
            labels_ids = [-100]
            input_ids_prefix = []
            attention_mask_prefix = []
            position_ids_prefix = []
            labels_ids_prefix = [-100]

            if int(self.prefix_length) == self.prefix_length:
                idx_with_grad = np.random.choice(
                    range(1, len(features[i]["input_ids"]) - 1),
                    min(int(self.prefix_length), len(features[i]["input_ids"]) - 2),
                    replace=False,
                ).tolist()
            else:
                idx_with_grad = np.random.choice(
                    range(1, len(features[i]["input_ids"]) - 2),
                    min(
                        int(self.prefix_length * (len(features[i]["input_ids"]) - 2)),
                        len(features[i]["input_ids"]) - 2
                    ),
                    replace=False,
                ).tolist()

            idx_with_grad = [0] + idx_with_grad
            idx_with_grad.append(len(features[i]["input_ids"]) - 1)

            for idx, id in enumerate(features[i]["input_ids"]):
                if idx in idx_with_grad:
                    input_ids_prefix.append(id)
                    attention_mask_prefix.append(features[i]["attention_mask"][idx])
                    position_ids_prefix.append(idx)
                    if idx + 1 < len(features[i]["labels"]):
                        labels_ids_prefix.append(features[i]["labels"][idx + 1])
                else:
                    input_ids.append(id)
                    attention_mask.append(features[i]["attention_mask"][idx])
                    position_ids.append(idx)
                    if idx + 1 < len(features[i]["labels"]):
                        labels_ids.append(features[i]["labels"][idx + 1])

            assert len(labels_ids_prefix) == len(input_ids_prefix), (len(labels_ids_prefix), len(input_ids_prefix))
            # in case the last token position of input_ids_prefix
            # is not the global last token position
            # remove the first prepended -100
            if len(labels_ids) > len(input_ids):
                labels_ids = labels_ids[1:]
            if len(labels_ids_prefix) > len(input_ids_prefix):
                labels_ids_prefix = labels_ids_prefix[1:]
            position_ids_all.append(position_ids)
            position_ids_prefix_all.append(position_ids_prefix)
            s = {
                "input_ids_prefix": input_ids_prefix,
                "attention_mask_prefix": attention_mask_prefix,
                "labels_ids_prefix": labels_ids_prefix,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels_ids,
            }
            sample_batch.append(s)
        batch = {k: [dic[k] for dic in sample_batch] for k in sample_batch[0]}
        # padding
        for k, v in batch.items():
            if k.endswith("prefix"):
                if int(self.prefix_length) == self.prefix_length:
                    batch[k] = [vv[: int(self.prefix_length)] for vv in v]
                    max_len = max(max([len(vv) for vv in v]), int(self.prefix_length))  # 2, want to have prefix with fixed size
                    batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in v])
                else:
                    batch[k] = [vv[: int(self.prefix_length * max([len(vv) for vv in v]))] for vv in v]
                    max_len = max(max([len(vv) for vv in v]), int(self.prefix_length * max([len(vv) for vv in v])))  # 2, want to have prefix with fixed size
                    batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in v])
            elif k == "label":
                batch[k] = torch.tensor(v)
            else:
                batch[k] = [vv[: self.max_length] for vv in v]
                max_len = max(max([len(vv) for vv in v]), 2)
                batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in v])
        for i in range(len(position_ids_all)):
            max_len = batch["input_ids"].shape[1]
            position_ids_all[i] = position_ids_all[i][:max_len]
            if len(position_ids_all[i]) < max_len:
                position_ids_all[i] = position_ids_all[i] + [0] * (
                    max_len - len(position_ids_all[i])
                )
            max_len = batch["input_ids_prefix"].shape[1]
            position_ids_prefix_all[i] = position_ids_prefix_all[i][:max_len]
            if len(position_ids_prefix_all[i]) < max_len:
                position_ids_prefix_all[i] = position_ids_prefix_all[i] + [0] * (
                    max_len - len(position_ids_prefix_all[i])
                )
        batch["position_ids_prefix"] = torch.tensor(
            position_ids_prefix_all, dtype=batch["input_ids"].dtype
        )
        batch["position_ids"] = torch.tensor(
            position_ids_all, dtype=batch["input_ids"].dtype
        )
        batch["labels"] = batch["labels_ids_prefix"]
        del batch["labels_ids_prefix"]
        del batch["labels_ids"]
        return batch


class LlamaConfigPrefix(LlamaConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id: int = 0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        prefix_length: int = 8,
        prefix_hidden_size: int = 4096,
        **kwargs: int,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.prefix_length = prefix_length
        self.prefix_hidden_size = prefix_hidden_size


@dataclass
class CausalLMOutputWithPastAndPrefix(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        hidden_states_prefix (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, prefix_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_prefix: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LlamaPrefixForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfigPrefix):
        super().__init__(config)
        self.model = LlamaPrefixModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.prefix_length: int = (
            config.prefix_length if hasattr(config, "prefix_length") else 0
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_ids_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.Tensor] = None,
        position_ids_prefix: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastAndPrefix]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_ids_prefix=input_ids_prefix,
            attention_mask_prefix=attention_mask_prefix,
            position_ids_prefix=position_ids_prefix,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        prefix_output = outputs[1]
        logits = self.lm_head(prefix_output)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndPrefix(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            hidden_states_prefix=outputs.hidden_states_prefix,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


@dataclass
class BaseModelOutputWithPastAndPrefix(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        last_hidden_state_prefix (`torch.FloatTensor` of shape `(batch_size, prefix_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model for the prefix tokens.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        hidden_states_prefix (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, prefix_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    last_hidden_state_prefix: torch.FloatTensor = None  # pyre-ignore
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_prefix: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """

    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _make_causal_mask_prefix(
    input_ids_shape: torch.Size,
    position_ids,
    position_ids_prefix,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """

    bsz, seq_len, pre_len = input_ids_shape
    mask = torch.full(
        (bsz, seq_len + pre_len, seq_len + pre_len),
        torch.finfo(dtype).min,
        device=device,
    )
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    mask_prefix = torch.gather(
        mask, 1, position_ids_prefix.unsqueeze(-1).repeat(1, 1, seq_len + pre_len)
    )
    mask = torch.gather(
        mask, 1, position_ids.unsqueeze(-1).repeat(1, 1, seq_len + pre_len)
    )
    mask_prefix = torch.gather(
        mask_prefix,
        2,
        torch.cat((position_ids, position_ids_prefix), axis=1)
        .unsqueeze(-2)
        .repeat(1, pre_len, 1),
    )
    mask = torch.gather(
        mask,
        2,
        torch.cat((position_ids, position_ids_prefix), axis=1)
        .unsqueeze(-2)
        .repeat(1, seq_len, 1),
    )

    return mask.unsqueeze(1), mask_prefix.unsqueeze(1)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask_prefix(
    mask: torch.Tensor,
    dtype: torch.dtype,
    seq_len: Optional[int] = None,
    pre_len: Optional[int] = None,
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = pre_len + seq_len if seq_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class LlamaPrefixModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfigPrefix
    """

    def __init__(self, config: LlamaConfigPrefix):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LlamaPrefixDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.prefix_length: int = (
            config.prefix_length if hasattr(config, "prefix_length") else 0
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        attention_mask_prefix,
        input_shape,
        position_ids,
        position_ids_prefix,
        inputs_embeds,
        past_key_values_length,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            (
                combined_attention_mask,
                combined_attention_mask_prefix,
            ) = _make_causal_mask_prefix(
                input_shape,
                position_ids,
                position_ids_prefix,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask_prefix(
                torch.cat((attention_mask, attention_mask_prefix), axis=1),
                inputs_embeds.dtype,
                seq_len=input_shape[-2],
                pre_len=input_shape[-1],
            ).to(inputs_embeds.device)
            expanded_attn_mask, expanded_attn_mask_prefix = (
                expanded_attn_mask[:, :, : input_shape[-2], :],
                expanded_attn_mask[:, :, input_shape[-2] :, :],
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
            combined_attention_mask_prefix = (
                expanded_attn_mask_prefix
                if combined_attention_mask_prefix is None
                else expanded_attn_mask_prefix + combined_attention_mask_prefix
            )

        return combined_attention_mask, combined_attention_mask_prefix

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_ids_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.Tensor] = None,
        position_ids_prefix: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prefix_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndPrefix]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if position_ids_prefix is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            print("`position_ids_prefix` is None, building with self.prefix_length=", self.prefix_length)
            if int(self.prefix_length) == self.prefix_length:
                position_ids_prefix = torch.arange(
                    past_key_values_length,
                    int(self.prefix_length) + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
            else:
                position_ids_prefix = torch.arange(
                    past_key_values_length,
                    input_ids_prefix.shape[1] + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
            position_ids_prefix = position_ids_prefix.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            with torch.no_grad():
                inputs_embeds = self.embed_tokens(input_ids)

        if prefix_embeds is None:
            prefix_embeds = self.embed_tokens(input_ids_prefix)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        if attention_mask_prefix is None:
            print("`attention_mask_prefix` is None, building with self.prefix_length=", self.prefix_length)
            if int(self.prefix_length) == self.prefix_length:
                attention_mask_prefix = torch.ones(
                    ((batch_size, int(self.prefix_length))), device=inputs_embeds.device
                )
            else:
                    attention_mask_prefix = torch.ones(
                    ((batch_size, prefix_embeds.shape[0])), device=inputs_embeds.device
                )

        attention_mask, attention_mask_prefix = self._prepare_decoder_attention_mask(
            attention_mask,
            attention_mask_prefix,
            (batch_size, seq_length, prefix_embeds.shape[1]),
            position_ids,
            position_ids_prefix,
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds
        hidden_states_prefix = prefix_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_prefix = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                all_hidden_states_prefix = all_hidden_states_prefix + (
                    hidden_states_prefix,
                )

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    hidden_states_prefix,
                    attention_mask_prefix,
                    position_ids_prefix,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    hidden_states_prefix=hidden_states_prefix,
                    attention_mask_prefix=attention_mask_prefix,
                    position_ids_prefix=position_ids_prefix,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            hidden_states_prefix = layer_outputs[1]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states_prefix = self.norm(hidden_states_prefix)
        with torch.no_grad():
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            all_hidden_states_prefix = all_hidden_states_prefix + (
                hidden_states_prefix,
            )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    hidden_states_prefix,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndPrefix(
            last_hidden_state=hidden_states,
            last_hidden_state_prefix=hidden_states_prefix,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            hidden_states_prefix=all_hidden_states_prefix,
            attentions=all_self_attns,
        )


class LlamaPrefixDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaPrefixAttention(config=config)
        self.mlp = LlamaPrefixMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        hidden_states_prefix: Optional[torch.LongTensor] = None,
        attention_mask_prefix: Optional[torch.Tensor] = None,
        position_ids_prefix: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        residual_prefix = hidden_states_prefix

        with torch.no_grad():
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states_prefix = self.input_layernorm(hidden_states_prefix)

        # Self Attention
        (
            hidden_states,
            hidden_states_prefix,
            self_attn_weights,
            present_key_value,
        ) = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            hidden_states_prefix=hidden_states_prefix,
            attention_mask_prefix=attention_mask_prefix,
            position_ids_prefix=position_ids_prefix,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        hidden_states_prefix = residual_prefix + hidden_states_prefix

        # Fully Connected
        residual = hidden_states
        residual_prefix = hidden_states_prefix

        with torch.no_grad():
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        
        hidden_states_prefix = self.post_attention_layernorm(hidden_states_prefix)
        hidden_states_prefix = self.mlp(hidden_states_prefix)
        hidden_states_prefix = residual_prefix + hidden_states_prefix

        outputs = (
            hidden_states,
            hidden_states_prefix,
        )

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaPrefixAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        hidden_states_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.FloatTensor] = None,
        position_ids_prefix: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        _, prefix_len, _ = hidden_states_prefix.size()

        with torch.no_grad():
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_prefix = self.q_proj(hidden_states_prefix)
        key_states_prefix = self.k_proj(hidden_states_prefix)
        value_states_prefix = self.v_proj(hidden_states_prefix)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        query_states_prefix = query_states_prefix.view(
            bsz, prefix_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states_prefix = key_states_prefix.view(
            bsz, prefix_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states_prefix = value_states_prefix.view(
            bsz, prefix_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        kv_prefix_len = key_states_prefix.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len + kv_prefix_len)
        with torch.no_grad():
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        query_states_prefix, key_states_prefix = apply_rotary_pos_emb(
            query_states_prefix, key_states_prefix, cos, sin, position_ids_prefix
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states_prefix = repeat_kv(key_states_prefix, self.num_key_value_groups)
        value_states_prefix = repeat_kv(value_states_prefix, self.num_key_value_groups)

        with torch.no_grad():

            attn_weights = torch.matmul(
                query_states,
                torch.cat(  # pyre-ignore
                    (key_states, key_states_prefix), axis=2  # pyre-ignore
                ).transpose(2, 3),
                # key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

        attn_weights_prefix = torch.matmul(
            query_states_prefix,
            torch.cat(  # pyre-ignore
                (key_states, key_states_prefix), axis=2  # pyre-ignore
            ).transpose(2, 3),
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (
            bsz,
            self.num_heads,
            q_len,
            kv_seq_len + kv_prefix_len,
        ):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len + kv_prefix_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        if attention_mask_prefix is not None:
            if attention_mask_prefix.size() != (
                bsz,
                1,
                kv_prefix_len,
                kv_seq_len + kv_prefix_len,
            ):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights_prefix = attn_weights_prefix + attention_mask_prefix

        # upcast attention to fp32
        with torch.no_grad():
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(
                attn_weights,
                torch.cat((value_states, value_states_prefix), axis=2)
                # value_states
            )

        attn_weights_prefix = nn.functional.softmax(
            attn_weights_prefix, dim=-1, dtype=torch.float32
        ).to(query_states_prefix.dtype)

        attn_output_prefix = torch.matmul(
            attn_weights_prefix, torch.cat((value_states, value_states_prefix), axis=2)
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        with torch.no_grad():
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output_prefix = attn_output_prefix.transpose(1, 2).contiguous()

        attn_output_prefix = attn_output_prefix.reshape(
            bsz, prefix_len, self.hidden_size
        )

        with torch.no_grad():
            attn_output = self.o_proj(attn_output)
        attn_output_prefix = self.o_proj(attn_output_prefix)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_output_prefix, attn_weights, past_key_value


class LlamaPrefixMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
