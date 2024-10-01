import logging
import math

logger = logging.getLogger(__name__)


def adopt_rope_config(config, model_args, data_args):
    """
    TODO: merge with patch_qwen2_rope_scaling
    ref: 
        - https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/
        - https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/model_utils/rope.py
    """
    if model_args.rope_scaling is None:
        logger.info("will not scale RoPE")
        return

    if model_args.rope_scaling_factor is not None:
        scaling_factor = model_args.rope_scaling_factor
    else:
        # calculate scaling factor
        scaling_factor = 1.0

        if data_args.max_seq_len:
            # curr_max_len: max position embedding base model supports
            curr_max_len = getattr(config, "max_position_embeddings", None)

            if curr_max_len:
                if data_args.max_seq_len > curr_max_len:
                    logger.info(
                        f"extend model max len to: {round(data_args.max_seq_len / 1000)}K"
                    )
                    setattr(
                        config, "max_position_embeddings", data_args.max_seq_len,
                    )
                    scaling_factor = float(
                        math.ceil(data_args.max_seq_len / curr_max_len)
                    )
                else:
                    logger.warning(
                        "curr model already support max len: "
                        f"{data_args.max_seq_len}"
                    )
    
    if config.model_type == "qwen2":
        if hasattr(model_args, "original_max_position_embeddings"):
            original_max_position_embeddings = model_args.original_max_position_embeddings
        else:
            original_max_position_embeddings = config.max_position_embeddings

        scaling_settings = {
            "type": model_args.rope_scaling, "factor": scaling_factor,
            "original_max_position_embeddings": original_max_position_embeddings,
        }

    else:
        if not hasattr(config, "rope_scaling"):
            logger.warning("curr model not support RoPE scaling")
            return 

        scaling_settings = {
            "type": model_args.rope_scaling, "factor": scaling_factor,
        }

    setattr(config, "rope_scaling", scaling_settings)
    logger.info(f"adopt scaling settings: {scaling_settings}")
