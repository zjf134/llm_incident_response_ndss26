# -*- coding: utf-8 -*-
"""
response_generation.py —— 步骤 (S3)：用微调好的 LLM 生成事故响应动作
====================================================================

【本脚本在论文中的位置】
  - 对应论文附录 F 的步骤 (S3)（Experiment E3），验证 Major Claim (C5)：
    "本方法能生成有效的事故响应计划"。论文第 20 页给出了期望输出示例。

【4-bit 适配说明（与原版的唯一实质差异）】
  * 原版以 fp16 加载需约 28 GB 显存；本仓库版改为 4-bit NF4 量化加载
    （论文 Table 7 的方案，计算精度 fp16 以兼容图灵卡），约 7.4 GB 即可运行。
  * 其余逻辑（自回归生成、温度采样、流式输出）与原版完全一致。

【本脚本演示的能力与论文的对应关系】
  1) 自回归生成（论文 §IV-B，式 (1)）：逐 token 采样直到结束符；
     temperature=0.8 保证输出多样性——这是 §IV-D 规划算法中
     "同一状态能采样出多个候选动作"的前提。
  2) CoT 推理 + 结构化输出（论文附录 B 的提示词模板）：
     <think>推理</think> + {"Action": "...", "Explanation": "..."}。

【重要说明：完整方法的三步在何处实现？（创新点 3 & 4）】
  * 第 2 步 RAG 与第 3 步前瞻规划（算法 1、命题 1/2）实现在配套库 llm_recovery
    与 CSLE 决策支持系统中，本脚本演示"单步动作生成"这一基础能力。

【运行前提】
  - 需要 GPU（约 8 GB 显存即可，4-bit 后 2080 Ti 级别单卡可跑）；
  - 无 GPU 时观看演示视频：https://www.youtube.com/watch?v=SCxq2ye-R4Y
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 服务器直连 huggingface.co 被墙，默认走国内镜像

import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading

if __name__ == '__main__':
    # 优先使用 GPU；4-bit 量化后单张 11 GB 卡即可容纳。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 4-bit NF4 + 双重量化（论文 Table 7），计算精度 fp16（图灵卡不支持 bf16）。
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "kimhammar/LLMIncidentResponse",
        device_map="auto",
        quantization_config=bnb,
    )
    tokenizer = AutoTokenizer.from_pretrained("kimhammar/LLMIncidentResponse")
    # 加载微调数据集（约 536MB，已缓存则秒开），随机抽取一条真实事故指令作为提示。
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="examples_16_june.json")
    instructions = dataset["train"]["instructions"][0]
    answers = dataset["train"]["answers"][0]
    model.eval()  # 推理模式：关闭 dropout 等训练期行为。
    instruction = random.choice(instructions)
    # 指令内容 = 论文附录 B 模板：系统描述 + 日志 + 事故摘要 + 六维状态 + 已执行动作 + 任务要求。
    inputs = tokenizer(instruction, return_tensors="pt").to(device)
    # 生成参数与原版一致：max_new_tokens=6000、temperature=0.8、随机采样
    # （温度>0 是规划算法"同一状态可采样出多个候选动作"的前提）。
    gen_kwargs = dict(
        max_new_tokens=6000,
        temperature=0.8,
        do_sample=True
    )
    # 流式输出：后台线程生成，主线程边收边打印。
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs={**inputs, **gen_kwargs, "streamer": streamer}
    )
    thread.start()
    for new_text in streamer:
        print(new_text, end="", flush=True)
