pkill -f "watch_resource"
pkill -f "spawn_main"

sleep 10  # wait above processes down

export TOKENIZERS_PARALLELISM=false

WORLD_SIZE=4
NUM_PROCESSES=8
MASTER_PORT=6678
NODE_NAME=`echo $ILOGTAIL_PODNAME | awk -F 'tfjob-' '{print $2}'`
NODE_NAME=${NODE_NAME:-master-0}

POD_NAME=${ILOGTAIL_PODNAME}

MASTER_IP_FILE="/experiment/llmx/debugging/dist/master_ip"


if [[ $POD_NAME == *"master"* ]]; then
    if [ -f "$MASTER_IP_FILE" ]; then
        rm -f "$MASTER_IP_FILE"
    fi

    NODE_RANK=0
    MASTER_IP=$(hostname -I | awk '{print $1}')
    echo "$MASTER_IP" > $MASTER_IP_FILE
    echo "Master Node: NODE_RANK = $NODE_RANK, MASTER_IP = $MASTER_IP"

else
    sleep 3  # wait master process writing its IP address
	if [ -f "$MASTER_IP_FILE" ]; then
		MASTER_IP=$(cat "$MASTER_IP_FILE")
	else
    	while [[ -z "$MASTER_IP_FILE" ]]
    	do
        	if [ -f "$MASTER_IP_FILE" ]; then
            	MASTER_IP=$(cat "$MASTER_IP_FILE")
				echo "master ip $MASTER_IP"
        	else
           	 	echo "waiting master write its IP"
            	sleep 1
        	fi
    	done
	fi

    NODE_RANK=$(echo $POD_NAME | grep -oP 'worker-\d+' | cut -d'-' -f2)
    NODE_RANK=$((NODE_RANK + 1))
    echo "Worker Node: NODE_RANK = $NODE_RANK, MASTER_IP = $MASTER_IP"
fi


# --==== Start training! ====--
python -m torch.distributed.run --nnode=$WORLD_SIZE --nproc_per_node=$NUM_PROCESSES --node_rank=$NODE_RANK --master_addr=$MASTER_IP --master_port=$MASTER_PORT ../task/run_train.py \
    --dataset_name risk \
    --model_name_or_path /modelhub/Qwen2_5-32B-Instruct \
    --chat_format qwen \
    --output_dir /checkpoints/exp00_nrank-$NODE_RANK \
    --save_total_limit 2 \
    --training_stage sft \
    --parameter_mode full \
    --lora_target q_proj,v_proj \
    --do_train true \
    --rope_scaling yarn \
    --max_seq_len 128000 \
    --lr_scheduler_type cosine \
    --sequence_parallel_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps 100 \
    --report_to tensorboard \
    --logging_steps 1 \
    --ddp_find_unused_parameters false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --overwrite_output_dir \
    --bf16 \
    --deepspeed ./ds_config/ds_z3.json


# -==== clear dist ipfile ====-
rm -f "$MASTER_IP_FILE"