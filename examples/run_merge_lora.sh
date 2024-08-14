export CUDA_VISIBLE_DEVICES=0

python ../task/run_merge_lora.py \
    --model_name_or_path /mnt/mashihan.msh/ptm/Qwen1.5-14B-Chat \
    --checkpoint_dir /mnt/mashihan.msh/experiment/llmx/debugging/0726_sft_test/checkpoint-75 \
    --merged_dir /mnt/mashihan.msh/experiment/llmx/debugging/0726_sft_test/merged