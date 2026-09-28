# -*- coding: utf-8 -*-
"""
load_training_dataset.py —— 步骤 (S2)：加载微调训练数据集
====================================================================

【本脚本在论文中的位置】
  - 对应论文附录 F 的步骤 (S2)（Experiment E2），验证 Major Claim (C1)：
    "发布首个面向事故响应的指令微调数据集"。

【数据集与论文的对应关系（创新点 2：68,000 条事故指令数据集）】
  论文 §IV-B "Dataset generation"（第 4 页）：
    * 论文定义指令-答案对 D = {(x^i, y^i)}，其中：
        - 指令 x^i：事故相关信息（系统描述 + 日志 + 当前恢复状态 + 已执行动作）+ 任务要求；
        - 答案 y^i：正确的响应动作，并附带链式思考（CoT）推理步骤。
    * 指令分两类：
        1) 动作生成指令（action-generation）—— 让 LLM 生成下一步响应动作；
        2) 状态预测指令（state-prediction）—— 让 LLM 判断当前响应进度
          （攻击是否被遏制/系统是否加固/取证是否完成等），
          这正对应论文 §IV-D 定义 1 的六维恢复状态 s_t = (遏制, 评估, 取证,
          驱逐, 加固, 恢复)，是该状态能被 LLM 预测的基础。
    * 数据构成：先在 64 台主机的 Docker 测试床（论文附录 A，Table 2/5）上
      真实执行侦察、爆破、漏洞利用攻击，得到 500 对真实数据；
      再用 Gemini 2.5 Pro 和 OpenAI O3 合成扩充到 68,000 对，
      覆盖多种攻击类型与系统架构（MITRE ATT&CK 战术分布见论文 Fig. 4，
      其中 INITIAL ACCESS / EXECUTION / EXFILTRATION 最常见）。

【数据托管位置】
  数据集托管在 Hugging Face：kimhammar/CSLE-IncidentResponse-V1
  （examples_16_june.json 约 536 MB；另有 Zenodo 存档 DOI 10.5281/zenodo.17459636）。
  首次运行自动下载到本地缓存 —— 代码在本仓库、数据在云端。

【运行前提】
  - 普通 CPU、8 GB 内存、约 1 GB 存储即可（论文 Table 8）。
"""

from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == '__main__':
    print("Loading training dataset.")
    # load_dataset(...)：从 Hugging Face Hub 拉取数据集（首次运行会下载约 536MB）。
    # data_files 指定数据文件；脚本用单文件 JSON 装载全部 68,000 条样本。
    # 注意：README 中的示例输出显示默认拉取的是 examples_16_june.json；
    # 这里传入 states_examples.json（状态预测类指令的示例文件）。
    # 若想加载动作生成类数据，可把 data_files 换成 "examples_16_june.json"。
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="states_examples.json")
    # 数据集以两个平行列表组织（每个元素是一条样本）：
    #   instructions[i] = 指令 x^i（系统描述+日志+事故摘要+状态+任务指令，
    #                     即论文附录 B 的提示词模板填充后的完整文本）
    #   answers[i]      = 答案 y^i（CoT 推理 + 最终动作，含 <think>...</think> 段）
    instructions = list(dataset["train"]["instructions"][0])
    answers = list(dataset["train"]["answers"][0])
    print(f"Training dataset loaded successfully. Number of instructions: {len(instructions)}. "
          f"Number of answers: {len(answers)}.")
    # 打印第一条指令样例，直观查看论文附录 B 提示词模板的实际填充效果：
    #   ### System: 系统描述 ... ### Logs: 日志 ... ### Incident: 事故摘要 ...
    #   ### State: 六维恢复状态 ... ### Previous recovery actions: 已执行动作 ...
    #   ### Instruction: 任务要求（生成下一步响应动作 / 预测恢复状态）
    print(instructions[0])
