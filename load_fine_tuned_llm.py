# -*- coding: utf-8 -*-
"""
load_fine_tuned_llm.py —— 步骤 (S1)：加载微调好的事故响应 LLM
====================================================================

【本脚本在论文中的位置】
  - 对应论文附录 F 的步骤 (S1)（论文第 19-20 页，Experiment E1/E2）。
  - 验证论文 Major Claims 中的 (C2)"首个开源的事故响应微调 LLM"
    和 (C3)"我们的 LLM 是轻量级的"。

【代码对应的论文核心思想（创新点 1：轻量化 + 领域微调）】
  论文 §IV-A / §IV-B（第 3-4 页）：
    * 与以往"提示工程 + 前沿大模型"（如 OpenAI O3、Gemini 2.5 Pro）的路线不同，
      本文把一个 **14B 参数的轻量模型**（DeepSeek-R1-Distill-Qwen-14B，
      论文 Remark 1：14B ≪ 671B 的 DeepSeek-R1）在 68,000 条事故数据上
      做指令微调（instruction fine-tuning），使其对齐事故响应的阶段与目标。
    * 微调带来的收益见论文 Fig. 6：动作准确率从 0.16 提升到 0.89（带 CoT 0.89），
      说明微调的主要作用是让模型学会训练数据中响应动作的"风格与规范"。

【本仓库只放代码、权重不进仓库的原因】
  微调权重托管在 Hugging Face（约 15 GB，含 14B 基座模型 + 201 MB LoRA 适配器），
  本脚本第一次运行时会自动下载到本地 Hugging Face 缓存目录。
  —— 这正是"本机只留代码、项目与数据上 GitHub/云端"的分工方式。

【运行前提】
  - Python 3.8+；普通 CPU 即可（论文 Table 8：CPU、72 GB 内存、100 GB 存储）。
  - 首次运行需下载约 15 GB 权重，依赖网络；本机存储紧张时可跳过本脚本，
    直接观看论文官方演示视频：https://www.youtube.com/watch?v=SCxq2ye-R4Y
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == '__main__':
    print("Loading the fine-tuned incident response LLM.")
    # AutoModelForCausalLM 会根据 Hugging Face 仓库中的配置自动构建因果语言模型。
    # "kimhammar/LLMIncidentResponse" = 论文作者发布在 Hugging Face 上的微调产物，
    #   包含：DeepSeek-R1-Distill-Qwen-14B 基座 + 事故响应 LoRA 适配器权重(约 201MB)。
    # device_map="auto"  ：由 accelerate 自动把模型层分配到 GPU/CPU，
    #                      本地无 GPU 时也能跑通加载流程（只是推理会很慢）。
    # torch_dtype=float16：半精度加载，显存/内存占用约减半（14B 模型约需 28 GB 内存）。
    #                      这也是论文强调"轻量化、可在商用硬件运行"的工程体现之一。
    model = AutoModelForCausalLM.from_pretrained(
        "kimhammar/LLMIncidentResponse",
        device_map="auto",
        torch_dtype=torch.float16
    )
    # 分词器：把指令文本切分成 token 序列 x = (x1, ..., xn)，供模型按
    # 论文式 (1) p_θ(x_{n+1} | x1, ..., xn) 自回归地逐 token 生成响应。
    tokenizer = AutoTokenizer.from_pretrained("kimhammar/LLMIncidentResponse")
    print(f"LLM loaded successfully on device: {model.device}")
