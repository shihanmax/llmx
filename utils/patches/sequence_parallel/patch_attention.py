import logging

from transformers.utils.import_utils import is_flash_attn_2_available

from .custom_models.qwen2 import qwen2_attn_forward


logger = logging.getLogger(__name__)


def patch_attention_for_qwen2(model, var_len_attn=False):

    def _patch_forward(module, qwen_attn_forward):
        new_forward = qwen_attn_forward.__get__(module, model.__class__)
        setattr(module, 'forward', new_forward)

    if not is_flash_attn_2_available():
        raise Exception("sequence parallel needs flash_attn_2!")

    model_name = model.__class__.__name__

    if "Qwen2" not in model_name:
        logger.warning(f"Model {model_name} not support sequence parallel")
        return
    
    # TODO: varlen attn not supported yet
    logger.info(f"patching attention module for {model_name}")

    for module in model.modules():
        if "Qwen2FlashAttention2" in type(module).__name__:
            _patch_forward(module, qwen2_attn_forward)
