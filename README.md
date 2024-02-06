# LLMX

开源大模型（LLM）微调工具，目前支持ChatGLM、百川、Llama、Qwen、Yi等开源大模型的预训练、微调和推理。


## 模型支持
| Model     | chat_format | lora target     |  train | inference |
|-----------|-----------|-----------------| -------|-----------|
| Baichuan2 | baichuan2 | W_pack          |  ✅     | ✅         |
| ChatGLM2  | chatglm2  | query_key_value |  ✅     | ✅         |
| ChatGLM3  | chatglm3  | query_key_value |  ✅     | ✅         |
| Qwen      | qwen      | c_attn          |  ✅     |             |
| Yi        | yi        | q_proj,v_proj   |  ✅     | ✅         |
| Llama2    | llama2    | q_proj,v_proj   |  ✅     | ✅         |


支持Chat models对应的base版本（请将参数`chat_format`参数设置为`base`）。

## 快速开始

**环境安装**

```bash
conda create -n llmx python==3.9 && conda activate llmx
git clone https://github.com/shihanmax/llmx.git
cd llmx && pip install -r requirements.txt
```

参考下文`训练数据格式`准备训练数据（目前支持sft（有监督微调）、pt（预训练）、dpo（直接偏好优化）），示例数据分别对应着`llmx/resource/data/`目录下的sft_demo、pt_demo、dpo_demo。

**SFT**

```bash 
bash ./scripts/run_sft.sh
```

```bash
accelerate config
accelerate launch ../runner/run_train.py \
    --dataset_name sft_demo \
    --model_name_or_path /path/to/plm_path \
    --output_dir ../debugging/checkpoint_path \
    --training_stage sft \
    --parameter_mode lora \
    --lora_target query_key_value \
    --max_seq_len 5000 \
    --do_train true \
    --learning_rate 5e-4 \
    --num_train_epochs 5.0 \
    --lr_scheduler_type cosine \
    --save_steps 10 \
    --report_to tensorboard \
    --logging_steps 2 \
    --ddp_find_unused_parameters false \
    --per_device_train batch size 2 \
    --gradient_accumulation_steps 4 \
    --overwrite_output_dir \
    --fp16
```


**PT**

```bash 
bash ./scripts/run_pt.sh
```

```bash
accelerate config
accelerate launch ../runner/run_train.py \
    --dataset_name pt_demo \
    --model_name_or_path /path/to/plm \
    --output_dir ../debugging/pt_test \
    --training_stage pt \
    --parameter_mode lora \
    --lora_target query_key_value \
    --max_seq_len 5000 \
    --do_train true \
    --learning_rate 5e-4 \
    --num_train_epochs 10.0 \
    --lr_scheduler_type cosine \
    --save_steps 10 \
    --report_to tensorboard \
    --logging_steps 2 \
    --ddp_find_unused_parameters false \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --overwrite_output_dir \
    --fp16
```


**DPO Training**
```bash 
bash ./scripts/run_dpo.sh
```

```bash
accelerate config
accelerate launch ../runner/run_train.py \
    --dataset_name dpo_demo \
    --model_name_or_path /path/to/plm \
    --chat_format chatglm2 \
    --output_dir ../debugging/_dpo_test \
    --training_stage dpo \
    --dpo_beta 0.1 \
    --dpo_loss_type sigmoid \
    --parameter_mode lora \
    --lora_target query_key_value \
    --max_seq_len 3000 \
    --do_train true \
    --learning_rate 5e-4 \
    --num_train_epochs 5.0 \
    --lr_scheduler_type cosine \
    --save_steps 100 \
    --report_to tensorboard \
    --logging_steps 10 \
    --ddp_find_unused_parameters false \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --remove_unused_columns false \
    --overwrite_output_dir \
    --fp16
```

**LoRA权重合并**
```bash 
bash ./scripts/run_merge_lora.sh
```

```bash
export CUDA_VISIBLE_DEVICES=0

python ../runner/run_merge_lora.py \
    --model_name_or_path /path/to/plm \
    --checkpoint_dir /path/to/ckpt/checkpoint-xxx \
    --merged_dir /path/to/merged
```


**多卡batch推理**
```bash 
bash ./scripts/run_predict.sh
```

```bash
python ../runner/run_inference.py \
    --model_name_or_path path/to/merged_model \
    --device_ids "0-8" \
    --chat_format baichuan2 \
    --predict_file_path path/to/predict_data.json \
    --save_predict_result_to path/to/predict_result.json \
    --per_device_predict_batch_size 12 \
    --predict_dry_run_size 0
```

**Chat（命令行）**
```bash 
bash ./scripts/run_chat.sh
```

```bash
python ../runner/run_generate.py \
  --model_name_or_path /path/to/plm \
  --chat_format chatglm2 \
  --topk 3 \
  --temperature 0.5
```


## 训练数据格式

### 预训练（pt）
```json
[
    {
        "raw": "《星际穿越》是一部探讨人类生存与宇宙奥秘的科幻电影。在地球面临生存危机的未来，一组勇敢的宇航员踏上了穿越虫洞的未知旅程，寻找可能的新家
        园。电影以其深邃的主题、壮丽的视觉效果和对时间与爱的哲学思考，赢得了观众和评论家的广泛赞誉。"
    }
]
```

### 有监督微调（sft）
```json
[
    {
        "instruction": "还有吗？",
        "input": "",
        "output": "当然！这是另外一首...",
        "history": [
            "有哪些关于雪的古诗词？",
            "柳宗元的《江雪》：\n千山鸟飞绝，万径人踪灭。\n孤舟蓑笠翁，独钓寒江雪。"
        ]  // `history` is optional
    }
]
```


### dpo训练
```json
[
    {
        "instruction": "介绍水的沸点",
        "input": "",
        "history": [],
        "chosen": "在标准大气压下，水的沸点为100℃(212°F)。当在更高的压力下加热时，水的沸点会升高。例如，水会在115℃(239°℉)的温度和1 bar的大气压力下沸腾。在更高的压力下，例如在海底经历的压力，水的沸点可高达374℃(705°℉)。",
        "rejected": "我不知道。"
    }
]

```

## Plans
| 类型            | 任务                                           | 状态 | 优先级 | 备注 |
|-----------------|------------------------------------------------|------|--------|------|
| pipeline优化    | 根据权重模型名，自动推断lora target，formatter |      | 2      |      |
| pipeline优化    | torch_dtype自动推断(model_loader ～L40)        |      | 3      |      |
| param-efficient | full                                           | ✅    |        |      |
| param-efficient | LoRA                                           | ✅    |        |      |
| param-efficient | LongLoRA                                       |      | 0      |      |
| param-efficient | QLoRA                                          |      | 0      |      |
| data            | 多轮history支持                                | ✅    |        |      |
| data            | non-chat模式支持（plain model）                | ✅    |        |      |
| data            | 收集benchmark数据集                            |      | 3      |      |
| bugifx          | demo（chat）bug（多轮对话session）             |      | 3      |      |
| 训练方法        | dpo                                            | ✅    |        |      |
| 训练方法        | pt                                             | ✅    |        |      |
| 训练方法        | sft                                            | ✅    |        |      |
| 训练方法        | ppo                                            |      | 1      |      |
| 训练方法        | rm                                             |      | 1      |      |
| 工具调用        | chatglm3工具调用支持                           |      | 3      |      |
| 并行训练        | 单机多卡                                       | ✅    |        |      |
| 并行训练        | 多机多卡                                       |      | 3      |      |
