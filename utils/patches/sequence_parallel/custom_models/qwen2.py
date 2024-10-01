# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import logging
import warnings
from typing import Optional

import torch
import torch.distributed as dist

from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb, repeat_kv,
)

from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward
)

from ..attention import (
    pre_process_for_sequence_parallel_attn, 
    post_process_for_sequence_parallel_attn,
)

from ..setup_dist import get_sequence_parallel_world_size, get_sequence_parallel_rank

from ..utils import get_rank

logger = logging.getLogger(__name__)

def flash_attention_support_window_size():
    try:
        from flash_attn import flash_attn_func
        _flash_supports_window_size = 'window_size' in list(
            inspect.signature(flash_attn_func).parameters)
    except ImportError:
        _flash_supports_window_size = None
    
    return _flash_supports_window_size


def debug_tensor(t, msg):  # TODO:debug
    print(f">>>\n{get_rank()}: {msg}: {t.shape}\n<<<")


def qwen2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    # logger.debug("in patched qwen2 atten forward")
    
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in '
            'v4.37. Please make sure use `attention_mask` instead.`')

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim,
    ).transpose(1, 2)

    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim,
    ).transpose(1, 2)

    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim,
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                'The cache structure has changed since version v4.36. '
                f'If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, '
                'please make sure to initialize the attention class '
                'with a layer index.')
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                       self.layer_idx)

    assert position_ids is not None
    rotary_seq_len = max(kv_seq_len, position_ids.max().item() + 1)
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    # use_sliding_windows = (
    #     flash_attention_support_window_size()
    #     and getattr(self.config, 'sliding_window', None) is not None
    #     and kv_seq_len > self.config.sliding_window
    #     and self.config.use_sliding_window
    # )

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value
        # `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, 'sliding_window', None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    'past key must have a shape of (`batch_size, num_heads, '
                    'self.config.sliding_window-1, head_dim`), got'
                    f' {past_key.shape}'
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones_like(attention_mask[:, -1:])
                    ], dim=-1,
                )

        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs,
        )

    # repeat k/v heads if n_kv_heads < n_heads for sequence parallel
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    sequence_parallel_enabled = (
        dist.is_initialized() and get_sequence_parallel_world_size() > 1
        and self.training
    )

    if sequence_parallel_enabled:
        query_states, key_states, value_states = \
            pre_process_for_sequence_parallel_attn(
                query_states, key_states, value_states,
            )

        # num_heads has been changed because of sequence parallel
        # `self.num_heads`` is not used in self._flash_attention_forward
        # in mistral/mixtral, we are doing this to avoid some unnecessary risk
        raw_num_head = self.num_heads
        self.num_heads = query_states.shape[-2]

    # logger.debug(f"here: {get_sequence_parallel_world_size()}, {get_sequence_parallel_rank()}")
    """
    for lower version transformers(<4.43) , set _flash_attention_forward to 
    self._flash_attention_forward
    """

    if (
        self.config.use_sliding_window
        and getattr(self.config, 'sliding_window', None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        # There may be bugs here, but we are aligned with Transformers
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_states.shape[1],
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    # logger.debug(f"done forward attn_output: {attn_output}\n: {get_sequence_parallel_world_size()}, {get_sequence_parallel_rank()}")

    if sequence_parallel_enabled:
        attn_output = post_process_for_sequence_parallel_attn(attn_output)
        self.num_heads = raw_num_head

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
