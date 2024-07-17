# # accelerate config
# accelerate launch ../runner/run_train.py \
#     --dataset_name insext \
#     --model_name_or_path /mnt/mashihan.msh/ptm/chatglm2-6b \
#     --chat_format chatglm2 \
#     --output_dir ../debugging/0716_sft_qlora_test \
#     --training_stage sft \
#     --parameter_mode lora \
#     --lora_target query_key_value \
#     --do_train true \
#     --max_seq_len 2048 \
#     --flash_attn \
#     --learning_rate 5e-4 \
#     --num_train_epochs 3.0 \
#     --lr_scheduler_type cosine \
#     --save_steps 1000 \
#     --report_to tensorboard \
#     --logging_steps 4 \
#     --ddp_find_unused_parameters false \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --overwrite_output_dir \
#     --fp16

python -m torch.distributed.run --nproc_per_node=4 ../runner/run_train.py \
    --dataset_name insext \
    --model_name_or_path /mnt/mashihan.msh/ptm/chatglm2-6b \
    --chat_format chatglm2 \
    --output_dir ../debugging/0716_sft_qlora_test \
    --training_stage sft \
    --parameter_mode lora \
    --lora_target query_key_value \
    --do_train true \
    --max_seq_len 2048 \
    --flash_attn \
    --learning_rate 5e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --save_steps 1000 \
    --report_to tensorboard \
    --logging_steps 4 \
    --ddp_find_unused_parameters false \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --overwrite_output_dir \
    --fp16 \
    --deepspeed ../resource/ds_config/ds_z1.json
