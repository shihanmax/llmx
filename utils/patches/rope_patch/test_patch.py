from transformers import Qwen2ForCausalLM, Qwen2Config
import torch
import sys
sys.path.append("..")

from rope_patch.qwen2_rope_patch import patch_qwen2_rope_scaling


# conf = Qwen2Config(
#     hidden_size=160,
#     intermediate_size=100,
#     num_hidden_layers=2,
#     num_attention_heads=8,
#     num_key_value_heads=1,
#     hidden_act="silu",
#     max_position_embeddings=8192,
#     initializer_range=0.02,
#     rms_norm_eps=1e-6,
#     use_cache=True,
#     tie_word_embeddings=False,
#     rope_theta=10000.0,
#     use_sliding_window=False,
#     max_window_layers=1,
#     attention_dropout=0.0,
#     rope_scaling={
#         "type": "dynamic-yarn",
#         "factor": 2.0,
#         "original_max_position_embeddings": 8192,
#     },
#     _attn_implementation="flash_attention_2",  # flash_attention_2
#     torch_dtype=torch.float16,
# )
conf = Qwen2Config.from_pretrained("./debug")
model = Qwen2ForCausalLM.from_pretrained(
    "./debug",
    config=conf,
    torch_dtype=torch.float16,
    # device_map=device_map,
    # **default_args,
)

# print(model.dtype)
model.to(torch.float16)

# patch_qwen2_rope_scaling(model)

model.to(torch.device("cuda"))

# model.save_pretrained("./debug/", 
# state_dict=model.state_dict(),
# )


input_ids = torch.randint(low=0, high=500, size=(3, 50), dtype=torch.long).to(torch.device("cuda"))

position_ids = torch.arange(0, 50, dtype=torch.long).to(torch.device("cuda")).unsqueeze(0).repeat(3, 1)

attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(torch.device("cuda"))
labels = torch.randint(low=0, high=500, size=(3, 50), dtype=torch.long).to(torch.device("cuda"))

res = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, labels=labels)
print(res)
print(model)