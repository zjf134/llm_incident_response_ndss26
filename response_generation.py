# -*- coding: utf-8 -*-
"""
response_generation.py —— 步骤 (S3)：用微调好的 LLM 生成事故响应动作
====================================================================

【本脚本在论文中的位置】
  - 对应论文附录 F 的步骤 (S3)（Experiment E3），验证 Major Claim (C5)：
    "本方法能生成有效的事故响应计划"。论文第 20 页给出了期望输出示例。

【本脚本演示的能力与论文的对应关系】
  1) 自回归生成（论文 §IV-B，式 (1)）：
       模型按 p_θ(x_{n+1} | x1, ..., xn) 逐 token 采样，直到产生
       结束符（EOS）。temperature=0.8 的随机采样对应论文中
       "采样生成 N 个候选动作"的能力来源——温度>0 使每次采样结果不同，
       这正是 §IV-D 规划算法中"一个状态能产生多个不同候选动作"的前提。

  2) CoT 推理 + 结构化输出（论文附录 B 的提示词模板）：
       微调后的模型输出形如：
           <think> ... 链式思考 ... </think>
           {"Action": "...", "Explanation": "..."}
       先推理、后给动作；动作必须是具体、可执行、不重复已执行动作的——
       这些约束全部写在微调数据的指令模板里（见 load_training_dataset.py）。

【重要说明：完整方法的三步在何处实现？（创新点 3 & 4）】
  论文 §IV-A 的方法 = 微调 + RAG + 决策论规划 三步：
    * 第 1 步 指令微调        —— 由本脚本加载的权重体现（训练过程见 fine_tune_llm.py）；
    * 第 2 步 RAG 信息检索     —— IOC 提取 → OTX 威胁情报 API 检索 → 拼接日志
                                 （论文 §IV-C，Fig. 7），实现在配套库 llm_recovery 中；
    * 第 3 步 前瞻规划（Alg.1）—— 采样 N 个候选动作，每个动作用 LLM 模拟 M 条
                                 恢复轨迹，估计期望恢复时间-to-go Q̃(s,a)，选
                                 argmin Q̃（论文式 (5)）；该"自我验证/self-consistency"
                                 机制把幻觉动作过滤掉，并给出幻觉概率上界
                                 （§V，Prop. 1 与 Prop. 2，其中
                                 Prop. 2：P(h ≥ h̄+ε) ≤ e^{-2ε²L}）。
                                 同样实现在 llm_recovery 库
                                 （https://github.com/Limmen/llm_recovery）
                                 与完整决策支持系统 CSLE
                                 （https://github.com/Limmen/csle）中。
  因此本脚本只演示"单步动作生成"这一基础能力；完整的"生成候选→模拟评估→
  选最短恢复时间"闭环请安装 llm_recovery 库或观看论文演示视频。

【运行前提】
  - 需要 GPU（论文 Table 8：RTX 8000 级别，72 GB 内存）。
  - 无 GPU 时可观看演示视频验证功能：https://www.youtube.com/watch?v=SCxq2ye-R4Y
  - 运行会触发下载模型（约 15GB）与数据集（约 536MB），存储紧张时注意磁盘空间。
"""

import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading

if __name__ == '__main__':
    # 优先使用 GPU（cuda），否则退回 CPU（会很慢，仅能验证流程）。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载微调模型与分词器（同 load_fine_tuned_llm.py，首次运行自动下载约 15GB 权重）。
    model = AutoModelForCausalLM.from_pretrained(
        "kimhammar/LLMIncidentResponse",
        dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("kimhammar/LLMIncidentResponse")
    # 加载微调数据集（约 536MB），从中随机抽取一条真实事故的指令作为提示。
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="examples_16_june.json")
    instructions = dataset["train"]["instructions"][0]
    answers = dataset["train"]["answers"][0]
    model.eval()  # 推理模式：关闭 dropout 等训练期行为。
    instruction = random.choice(instructions)
    # 把指令文本编码成 token 序列并搬到计算设备。
    # 该指令内容 = 论文附录 B 模板：系统描述 + Snort 日志 + 事故摘要 +
    # 当前六维恢复状态 + 已执行动作 + 任务指令（生成下一步响应动作）。
    inputs = tokenizer(instruction, return_tensors="pt").to(device)
    # 生成参数：
    #   max_new_tokens=6000 —— 覆盖 CoT 推理 + JSON 动作的总长度上限；
    #   temperature=0.8     —— 采样温度，>0 保证输出多样性（论文规划算法
    #                          依赖"同一状态可采样出多个候选动作"）；
    #   do_sample=True      —— 启用随机采样（对应论文式 (1) 的采样式生成）。
    gen_kwargs = dict(
        max_new_tokens=6000,
        temperature=0.8,
        do_sample=True
    )
    # 流式输出：TextIteratorStreamer 在后台线程中运行 model.generate，
    # 主线程迭代读取新生成的文本，实现"边生成边打印"。
    # skip_special_tokens=True 会隐藏 <think> 等特殊标记以外的控制符；
    # skip_prompt=True 不回显输入提示。
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs={**inputs, **gen_kwargs, "streamer": streamer}
    )
    thread.start()
    for new_text in streamer:
        print(new_text, end="", flush=True)
