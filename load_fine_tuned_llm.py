# -*- coding: utf-8 -*-
"""
load_fine_tuned_llm.py —— 步骤 (S1)：加载微调好的事故响应 LLM
====================================================================

【本脚本在论文中的位置】
  - 对应论文附录 F 的步骤 (S1)（Experiment E1/E2）。
  - 验证论文 Major Claims 中的 (C2)"首个开源的事故响应微调 LLM"
    和 (C3)"我们的 LLM 是轻量级的"。

【4-bit 适配说明（本仓库版与原版的唯一实质差异）】
  * 原版以 fp16 加载，14B 模型约需 28 GB 显存（论文用 RTX 8000 48GB 级别）；
  * 本仓库版改用论文表 7 的量化方案：4-bit NF4 + 双重量化加载，
    仅需约 7.4 GB，2×11 GB 的 RTX 2080 Ti（或任何 ≥8 GB 显存的设备）即可运行；
  * 模型本身不变：仍是 DeepSeek-R1-Distill-Qwen-14B 基座 + 事故响应 LoRA 适配器
    （约 201 MB），4-bit 只是权重存储位宽（QLoRA 标准做法，质量几乎无损）；
  * 计算精度用 fp16：图灵架构显卡（2080 Ti/T4）不支持 bf16。

【代码对应的论文核心思想（创新点 1：轻量化 + 领域微调）】
  论文 §IV-A / §IV-B（第 3-4 页）：
    * 与以往"提示工程 + 前沿大模型"的路线不同，本文把 14B 参数的轻量模型在
      68,000 条事故数据上做指令微调，使其对齐事故响应的阶段与目标；
    * 微调收益见论文 Fig. 6：动作准确率从 0.16 提升到 0.89。

【运行前提】
  - Python 3.8+；普通 CPU 即可完成加载（GPU 仅推理时需要）；
  - 首次运行需下载约 15 GB 权重（基座 + 适配器）到 Hugging Face 缓存；
    国内网络建议先 export HF_ENDPOINT=https://hf-mirror.com。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

if __name__ == '__main__':
    print("Loading the fine-tuned incident response LLM.")
    # 4-bit NF4 + 双重量化（论文 Table 7 的量化方案）；
    # 计算精度 fp16（图灵卡不支持 bf16）——加载后权重约 7.4 GB。
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # "kimhammar/LLMIncidentResponse" = 论文作者发布的微调产物：
    #   DeepSeek-R1-Distill-Qwen-14B 基座 + 事故响应 LoRA 适配器（约 201MB）。
    # device_map="auto"：多卡时自动把 48 层拆分到各卡；单卡则整卡装载。
    model = AutoModelForCausalLM.from_pretrained(
        "kimhammar/LLMIncidentResponse",
        device_map="auto",
        quantization_config=bnb,
    )
    # 分词器：把指令文本切分成 token 序列，供模型按论文式 (1) 自回归生成。
    tokenizer = AutoTokenizer.from_pretrained("kimhammar/LLMIncidentResponse")
    print(f"LLM loaded successfully on device: {model.device}")
