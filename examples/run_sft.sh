# accelerate config
# accelerate launch ../task/run_train.py \
#     --dataset_name insext \
#     --model_name_or_path /mnt/mashihan.msh/ptm/Qwen1.5-14B-Chat \
#     --chat_format qwen \
#     --output_dir ../debugging/0724_sft_test \
#     --training_stage sft \
#     --parameter_mode lora \
#     --lora_target q_proj,v_proj \
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
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --overwrite_output_dir \
#     --fp16

python -m torch.distributed.run --nproc_per_node=2 ../task/run_train.py \
    --dataset_name sft_demo \
    --model_name_or_path /mnt/mashihan.msh/ptm/Qwen1.5-14B-Chat \
    --chat_format qwen \
    --output_dir ../debugging/0726_sft_test \
    --training_stage sft \
    --parameter_mode lora \
    --lora_target q_proj,v_proj \
    --do_train true \
    --max_seq_len 1024 \
    --flash_attn \
    --learning_rate 5e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --save_steps 60 \
    --report_to tensorboard \
    --logging_steps 4 \
    --ddp_find_unused_parameters false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir \
    --fp16
    # --deepspeed ../resource/ds_config/ds_z1.json
