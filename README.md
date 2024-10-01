# LLMX

开源大模型（LLM）微调工具，目前支持 Qwen/Qwen1.5/Qwen2.5、ChatGLM2/3、Baichuan、Llama、Yi等开源大模型的预训练、微调和推理。


支持：
- LoRA、QLoRA
- 通过开启Sequence parallel（序列并行）以支持高达 256K 序列长度的全参数微调
- 通过 RoPE scaling 进行序列长度扩展
- VLLM 推理部署


## 模型支持
| Model         | chat_format | lora target     | train | inference |
|---------------|-------------|-----------------|-------|-----------|
| Baichuan2     | baichuan2   | W_pack          | ✅     | ✅         |
| ChatGLM2      | chatglm2    | query_key_value | ✅     | ✅         |
| ChatGLM3      | chatglm3    | query_key_value | ✅     | ✅         |
| Qwen          | qwen        | c_attn          | ✅     | ✅         |
| Qwen2/Qwen2.5 | qwen        | q_proj,v_proj   | ✅     | ✅         |
| Yi            | yi          | q_proj,v_proj   | ✅     | ✅         |
| Llama2        | llama2      | q_proj,v_proj   | ✅     | ✅         |


支持Chat models对应的base版本（请将参数`chat_format`参数设置为`base`）。


**版本要求**

| features          | requirements      | ref.                                                             |
|-------------------|-------------------|------------------------------------------------------------------|
| QLoRA             | CUDA>=11.2        |                                                                  |
| vllm              | CUDA==11.8,12.1   | https://docs.vllm.ai/en/latest/getting_started/installation.html |
| sequence parallel | flash-attn>=2.1.0 |                                                                  |

## 快速开始

**依赖安装**

```bash
conda create -n llmx python==3.9 && conda activate llmx
git clone https://github.com/shihanmax/llmx.git
cd llmx && pip install -r requirements.txt
```

参考下文`训练数据格式`准备训练数据（目前支持sft（有监督微调）、pt（预训练）、dpo（直接偏好优化）），示例数据分别对应着`llmx/resource/data/`目录下的sft_demo、pt_demo、dpo_demo。

**SFT（单机多卡）**

```bash 
bash ./scripts/run_sft.sh
```

**SFT（多机多卡）**

1. 建立各节点之间的 ssh 通信
2. 在各节点上运行以下命令

```bash 
bash ./scripts/run_sft_multi_node.sh
```

**PT**

```bash 
bash ./scripts/run_pt.sh
```

**DPO Training**
```bash 
bash ./scripts/run_dpo.sh
```

**LoRA权重合并**
```bash 
bash ./scripts/run_merge_lora.sh
```

**多卡batch推理**
```bash 
bash ./scripts/run_predict.sh
```


**vllm 推理** 

启动 vllm 服务
```bash 
bash ./scripts/serve_vllm_openai.sh
```

**Chat（命令行）**
```bash 
bash ./scripts/run_chat.sh
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
