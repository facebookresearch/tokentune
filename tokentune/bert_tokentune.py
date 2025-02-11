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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertConfig,
    BertEmbeddings,
    BertIntermediate,
    BertPreTrainedModel,
)

from transformers.models.bert.configuration_bert import BertConfig  # noqa
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging, ModelOutput, PaddingStrategy

logger = logging.get_logger(__name__)  # pyre-ignore[5]


"""
In this file, we apply the proposed TokenTune method to the BERT model implementation from the Transformers library.
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
    prefix_length: int = 16

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        sample_batch = []
        position_ids_all = []
        position_ids_prefix_all = []

        for i in range(len(features)):

            input_ids = []
            attention_mask = []
            token_type_ids = []
            position_ids = []

            input_ids_prefix = []
            attention_mask_prefix = []
            token_type_ids_prefix = []
            position_ids_prefix = []

            idx_with_grad = [0] + np.random.choice(
                range(1, len(features[i]["input_ids"])),
                min(self.prefix_length - 1, len(features[i]["input_ids"]) - 1),
                replace=False,
            ).tolist()

            for idx, id in enumerate(features[i]["input_ids"]):
                if idx in idx_with_grad:
                    input_ids_prefix.append(id)
                    attention_mask_prefix.append(features[i]["attention_mask"][idx])
                    token_type_ids_prefix.append(features[i]["token_type_ids"][idx])
                    position_ids_prefix.append(idx)
                else:
                    input_ids.append(id)
                    attention_mask.append(features[i]["attention_mask"][idx])
                    token_type_ids.append(features[i]["token_type_ids"][idx])
                    position_ids.append(idx)

            position_ids_all.append(position_ids)
            position_ids_prefix_all.append(position_ids_prefix)
            s = {
                "input_ids_prefix": input_ids_prefix,
                "attention_mask_prefix": attention_mask_prefix,
                "token_type_ids_prefix": token_type_ids_prefix,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            if "label" in features[i]:
                s["label"] = features[i]["label"]
            sample_batch.append(s)

        batch = {k: [dic[k] for dic in sample_batch] for k in sample_batch[0]}

        for k, v in batch.items():
            if k.endswith("prefix"):
                batch[k] = [vv[: self.prefix_length] for vv in v]
                max_len = max(max([len(vv) for vv in v]), 2)
                batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in v])
            elif k == "label":
                batch[k] = torch.tensor(v)
            else:
                batch[k] = [vv[: self.max_length] for vv in v]
                max_len = max(max([len(vv) for vv in v]), 2)
                batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in v])

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

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
        return batch


class BertConfigPrefix(BertConfig):

    model_type = "bert"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[bool] = None,
        prefix_length: int = 8,
        prefix_hidden_size: int = 512,
        use_input_as_prefix: bool = False,
        temperature: float = 0.1,
        **kwargs: int,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.prefix_length = prefix_length
        self.prefix_hidden_size = prefix_hidden_size
        self.temperature = temperature
        self.use_input_as_prefix = use_input_as_prefix


class BertPrefixForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config: BertConfigPrefix) -> None:
        super().__init__(config)
        self.num_labels: int = config.num_labels
        self.config = config

        self.bert = BertPrefixModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_ids_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.Tensor] = None,
        token_type_ids_prefix: Optional[torch.Tensor] = None,
        position_ids_prefix: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            input_ids_prefix=input_ids_prefix,
            attention_mask_prefix=attention_mask_prefix,
            token_type_ids_prefix=token_type_ids_prefix,
            position_ids_prefix=position_ids_prefix,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[2]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsAndPrefix(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        last_hidden_state_prefix (`torch.FloatTensor` of shape `(batch_size, prefix_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model for the prefix tokens.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
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
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None  # pyre-ignore
    last_hidden_state_prefix: torch.FloatTensor = None  # pyre-ignore
    pooler_output: torch.FloatTensor = None  # pyre-ignore
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_prefix: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertPrefixPooler(nn.Module):
    def __init__(self, config: BertConfigPrefix) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.use_input_as_prefix: bool = (
            hasattr(config, "use_input_as_prefix") and config.use_input_as_prefix
        )
        if not self.use_input_as_prefix:
            self.gate = nn.Parameter(torch.zeros(1))
        # self.gate = nn.Parameter(torch.Tensor([-1]))
        self.T: float = config.temperature

    def forward(
        self, hidden_states: torch.Tensor, hidden_states_prefix: torch.Tensor
    ) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        if self.use_input_as_prefix:
            if self.training:
                hidden_states = hidden_states_prefix.mean(1)
            else:
                hidden_states = torch.cat(  # pyre-ignore
                    (hidden_states_prefix, hidden_states), axis=1
                ).mean(1)
        else:
            first_token_tensor = hidden_states[:, 0]
            first_token_tensor_prefix = hidden_states_prefix.mean(1)

            gate = torch.sigmoid(self.gate / self.T)
            hidden_states = (
                gate * first_token_tensor_prefix + (1 - gate) * first_token_tensor
            )

        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPrefixModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(
        self, config: BertConfigPrefix, add_pooling_layer: bool = True
    ) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings: BertEmbeddings = BertEmbeddings(config)
        self.encoder = BertPrefixEncoder(config)

        self.pooler: Optional[BertPrefixPooler] = (
            BertPrefixPooler(config) if add_pooling_layer else None
        )

        self.prefix_length: int = (
            config.prefix_length if hasattr(config, "prefix_length") else 0
        )

        self.use_input_as_prefix: bool = (
            hasattr(config, "use_input_as_prefix") and config.use_input_as_prefix
        )
        if not self.use_input_as_prefix:
            self.prefix: nn.parameter.Parameter = nn.Parameter(
                torch.rand((config.prefix_length, config.hidden_size))
            )
            self.prefix.data.normal_(mean=0.0, std=config.initializer_range)


        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value) -> None:  # pyre-ignore
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_ids_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.Tensor] = None,
        token_type_ids_prefix: Optional[torch.Tensor] = None,
        position_ids_prefix: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentionsAndPrefix
    ]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
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

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = (
            input_ids.device
            if input_ids is not None
            else inputs_embeds.device  # pyre-ignore
        )

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape  # pyre-ignore
        )
        if attention_mask_prefix is None:
            attention_mask_prefix = torch.ones(
                ((batch_size, self.prefix_length)), device=device
            )

        if input_ids_prefix is not None:
            prefix_size = input_ids_prefix.size()
        else:
            prefix_size = (batch_size, self.prefix_length)
        extended_attention_mask_prefix = self.get_extended_attention_mask(
            attention_mask_prefix, prefix_size  # pyre-ignore
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        with torch.no_grad():
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

            # prefix_tokens = embedding_output.mean(axis=1, keepdim=True).expand(
            #     -1, self.config.prefix_length, -1
            # )

        if self.use_input_as_prefix:
            prefix_tokens = self.embeddings(
                input_ids=input_ids_prefix,
                position_ids=position_ids_prefix,
                token_type_ids=token_type_ids_prefix,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            prefix_tokens = embedding_output.mean(axis=1, keepdim=True).expand(
                -1, self.config.prefix_length, -1
            )

        encoder_outputs = self.encoder(
            embedding_output,
            prefix_tokens,
            attention_mask=extended_attention_mask,
            attention_mask_prefix=extended_attention_mask_prefix,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        prefix_output = encoder_outputs[1]
        pooled_output = (
            self.pooler(sequence_output, prefix_output)
            if self.pooler is not None
            else None
        )

        if not return_dict:
            return (sequence_output, prefix_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentionsAndPrefix(
            last_hidden_state=sequence_output,
            last_hidden_state_prefix=prefix_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            hidden_states_prefix=encoder_outputs.hidden_states_prefix,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@dataclass
class BaseModelOutputWithPastAndCrossAttentionsAndPrefix(ModelOutput):
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
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None  # pyre-ignore
    last_hidden_state_prefix: torch.FloatTensor = None  # pyre-ignore
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_prefix: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertPrefixEncoder(nn.Module):
    def __init__(self, config: BertConfigPrefix) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertPrefixLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_prefix: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask_prefix: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentionsAndPrefix]:
        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_prefix = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                all_hidden_states_prefix = all_hidden_states_prefix + (
                    hidden_states_prefix,
                )

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(
                    module: nn.Module,
                ) -> Callable[[Tuple[torch.Tensor]], Tuple[torch.Tensor]]:
                    def custom_forward(  # pyre-ignore
                        *inputs: Tuple[torch.Tensor],
                    ) -> Tuple[torch.Tensor]:
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    hidden_states_prefix,
                    attention_mask,
                    attention_mask_prefix,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    i,
                    hidden_states,
                    hidden_states_prefix,
                    attention_mask,
                    attention_mask_prefix,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            hidden_states_prefix = layer_outputs[1]
            batch_size = hidden_states_prefix.shape[0]
            # add prompt to every layer
            # hidden_states_prefix = hidden_states_prefix + self.prefixes[i].unsqueeze(
            #     0
            # ).expand(batch_size, -1, -1)
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_hidden_states_prefix = all_hidden_states_prefix + (
                hidden_states_prefix,
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    hidden_states_prefix,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentionsAndPrefix(
            last_hidden_state=hidden_states,
            last_hidden_state_prefix=hidden_states_prefix,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            hidden_states_prefix=all_hidden_states_prefix,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPrefixLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward: int = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention: BertPrefixAttention = BertPrefixAttention(config)
        self.is_decoder: bool = config.is_decoder
        self.add_cross_attention: bool = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention: BertAttention = BertAttention(
                config, position_embedding_type="absolute"
            )
        self.intermediate: BertIntermediate = BertIntermediate(config)
        # self.output: BertOutput = BertOutput(config)
        self.output: BertPrefixOutput = BertPrefixOutput(config)

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        hidden_states_prefix: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask_prefix: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            layer_idx,
            hidden_states,
            hidden_states_prefix,
            attention_mask,
            attention_mask_prefix,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        attention_output_prefix = self_attention_outputs[1]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output, layer_output_prefix = apply_chunking_to_forward(
            self.feed_forward_chunk,  # pyre-ignore
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
            attention_output_prefix,
            layer_idx,
        )
        outputs = (
            layer_output,
            layer_output_prefix,
        ) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)  # pyre-ignore

        return outputs

    def feed_forward_chunk(
        self,
        attention_output: torch.Tensor,
        attention_output_prefix: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        intermediate_output_prefix = self.intermediate(attention_output_prefix)
        layer_output_prefix = self.output(
            intermediate_output_prefix,
            attention_output_prefix,  # layer_idx == 0
        )
        return layer_output, layer_output_prefix  # pyre-ignore


class BertPrefixOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        use_skip_connection: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if use_skip_connection:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPrefixAttention(nn.Module):
    def __init__(
        self, config: BertConfig, position_embedding_type: Optional[str] = None
    ) -> None:
        super().__init__()
        self.self: BertSelfPrefixAttention = BertSelfPrefixAttention(
            config, position_embedding_type=position_embedding_type
        )
        # self.output: BertSelfOutput = BertSelfOutput(config)
        self.output: BertSelfPrefixOutput = BertSelfPrefixOutput(config)
        self.pruned_heads: Set[int] = set()

    def prune_heads(self, heads: List[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(  # pyre-ignore
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        hidden_states_prefix: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask_prefix: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            layer_idx,
            hidden_states,
            hidden_states_prefix,
            attention_mask,
            attention_mask_prefix,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        with torch.no_grad():
            attention_output = self.output(self_outputs[0], hidden_states)
        attention_output_prefix = self.output(
            self_outputs[1],
            hidden_states_prefix,  # layer_idx == 0
        )
        outputs = (attention_output, attention_output_prefix,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class BertSelfPrefixOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        use_skip_connection: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if use_skip_connection:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertSelfPrefixAttention(nn.Module):
    def __init__(
        self, config: BertConfig, position_embedding_type: Optional[str] = None
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads: int = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type: str = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings: int = config.max_position_embeddings
            self.distance_embedding: nn.Embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder: bool = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        hidden_states_prefix: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask_prefix: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            mixed_query_layer = self.query(hidden_states)
        mixed_query_layer_prefix = self.query(hidden_states_prefix)
        # mixed_query_layer_prefix += self.query_prefix(hidden_states_prefix)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)  # pyre-ignore
            value_layer = torch.cat(
                [past_key_value[1], value_layer], dim=2  # pyre-ignore
            )
        else:
            with torch.no_grad():
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer_prefix = self.transpose_for_scores(self.key(hidden_states_prefix))
            value_layer_prefix = self.transpose_for_scores(
                self.value(hidden_states_prefix)
            )

        with torch.no_grad():
            query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer_prefix = self.transpose_for_scores(mixed_query_layer_prefix)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)  # pyre-ignore

        # Take the dot product between "query" and "key" to get the raw attention scores.
        with torch.no_grad():
            attention_scores = torch.matmul(
                query_layer,
                torch.cat(  # pyre-ignore
                    (key_layer, key_layer_prefix), axis=2  # pyre-ignore
                ).transpose(-1, -2),
            )
        attention_scores_prefix = torch.matmul(
            query_layer_prefix,
            torch.cat(  # pyre-ignore
                (key_layer, key_layer_prefix), axis=2  # pyre-ignore
            ).transpose(-1, -2),
        )

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = (
                query_layer.shape[2],
                key_layer.shape[2],  # pyre-ignore
            )
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        with torch.no_grad():
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores_prefix = attention_scores_prefix / math.sqrt(
            self.attention_head_size
        )

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # attention_scores = attention_scores + attention_mask
            attention_scores = attention_scores + torch.cat(  # pyre-ignore
                (attention_mask, attention_mask_prefix), axis=-1
            )
        if attention_mask_prefix is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # attention_scores_prefix = attention_scores_prefix + attention_mask_prefix
            attention_scores_prefix = (
                attention_scores_prefix
                + torch.cat(  # pyre-ignore
                    (attention_mask, attention_mask_prefix), axis=-1
                )
            )

        # Normalize the attention scores to probabilities.
        with torch.no_grad():
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs_prefix = nn.functional.softmax(attention_scores_prefix, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with torch.no_grad():
            attention_probs = self.dropout(attention_probs)
        attention_probs_prefix = self.dropout(attention_probs_prefix)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        with torch.no_grad():
            context_layer = torch.matmul(
                attention_probs,
                torch.cat((value_layer, value_layer_prefix), axis=2),  # pyre-ignore
            )
        context_layer_prefix = torch.matmul(
            attention_probs_prefix,
            torch.cat((value_layer, value_layer_prefix), axis=2),  # pyre-ignore
        )

        with torch.no_grad():
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
        context_layer_prefix = context_layer_prefix.permute(0, 2, 1, 3).contiguous()
        new_context_layer_prefix_shape = context_layer_prefix.size()[:-2] + (
            self.all_head_size,
        )
        context_layer_prefix = context_layer_prefix.view(new_context_layer_prefix_shape)

        outputs = (
            (context_layer, context_layer_prefix, attention_probs)
            if output_attentions
            else (
                context_layer,
                context_layer_prefix,
            )
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs  # pyre-ignore
