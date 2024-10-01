from .rotary_embeddings import (
    RotaryEmbedding, LinearScalingRotaryEmbedding, DynamicNTKScalingRotaryEmbedding,
    YaRNScaledRotaryEmbedding, DynamicYaRNScaledRotaryEmbedding
)


def init_rope(self, config):

    if not hasattr(self.config, "rope_scaling") or self.config.rope_scaling is None:
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    else:
        scaling_type = config.rope_scaling["type"]
        scaling_factor = config.rope_scaling["factor"]

        if scaling_type == "linear":
            self.rotary_emb = LinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )

        elif scaling_type == "dynamic":
            self.rotary_emb = DynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )

        elif scaling_type == "yarn":
            original_max_position_embeddings = self.config.rope_scaling["original_max_position_embeddings"]
            self.rotary_emb = YaRNScaledRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scale=scaling_factor,
                original_max_position_embeddings=original_max_position_embeddings
            )
            
        elif scaling_type == "dynamic-yarn":
            original_max_position_embeddings = self.config.rope_scaling["original_max_position_embeddings"]
            self.rotary_emb = DynamicYaRNScaledRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                original_max_position_embeddings=original_max_position_embeddings
            )

        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


def patch_qwen2_rope_scaling(model, config):
    """Replace RoPE instance of Qwen2*Attention modules
    
    this method should be executed after the Qwen2ForCausalLM is instantiated.
    """

    QWEN2_ATTN_MODULE_NAMES = ["Qwen2Attention", "Qwen2FlashAttention2", "Qwen2SdpaAttention"]
    
    for module in model.modules():
        if type(module).__name__ in QWEN2_ATTN_MODULE_NAMES:
            init_rope(module, config)
