# -*- coding: utf-8 -*-
"""
fine_tune_llm.py —— 步骤 (S4)：在事故响应数据集上微调 LLM
====================================================================

【本脚本在论文中的位置】
  - 对应论文附录 F 的步骤 (S4)（Experiment E4）。
  - 为便于快速验证，脚本默认只用数据集的前 5 条样本微调（缩小版实验）；
    论文的完整微调用全部 68,000 条数据，在 4×RTX 8000 上训练约 800 分钟
    （论文 Fig. 5 的损失曲线），完整复现需数天。

【代码对应的论文核心内容（创新点 1 的训练侧实现）】
  1) 监督微调目标（论文 §IV-B，式 (2)）：
       L = -1/M * Σ_i Σ_k ln p_θ(y^i_k | x^i, y^i_1, ..., y^i_{k-1})
       即对答案 y^i 逐 token 计算交叉熵损失（teacher forcing：以真实答案的
       前 k-1 个 token 为条件预测第 k 个 token）。这一训练目标由
       llm_recovery 库中 LORA.supervised_fine_tuning() 内部的 Trainer 循环实现。

  2) LoRA 参数高效微调（论文 Table 7 超参数）：
       * rank r=64, alpha=128, dropout=0.05, learning rate=0.00095
         —— 与论文 Table 7 中 "LoRA rank r, α, dropout, learning rate =
         64, 128, 0.05, 0.00095" 完全一致；
       * LoRA 只训练注入的低秩适配矩阵（本例约 5033 万可训练参数，
         运行时输出 "Trainable parameters: 50331648"），
         相比 14B 全参数微调节省约 300 倍显存 —— 这是"轻量化可复现"
         的关键工程手段；
       * 论文 Fig. 5 对比了 0.00095（高）与 0.000095（低）两种学习率：
         高学习率收敛到更低损失，故本脚本采用 0.00095。

  3) 两类指令混合训练（论文 §IV-B）：
       数据集中的指令同时包含"动作生成"与"状态预测"两类，
       使模型既能生成响应动作，又能预测六维恢复状态 s_t（定义 1）——
       后者是规划算法（Alg. 1）中"用 LLM 模拟恢复轨迹"的基础。

【创新点小结：为什么说这篇论文的微调是"创新点"而非常规操作？】
  * 这是**首个面向事故响应领域的指令微调 LLM**（论文 Major Claim C1/C2），
    之前的工作全部依赖前沿模型的提示工程；
  * 数据集 = 64 主机测试床真实攻击日志（500 对）+ 前沿模型合成扩充
    （至 68,000 对）的混合策略（§IV-B Dataset generation）；
  * 微调显式对齐 MITRE D3FEND 六阶段恢复状态（定义 1），让 LLM 的输出
    可以被形式化为随机最短路径问题（§IV-D），从而支撑后续的
    前瞻规划与幻觉概率理论分析（§V，Prop. 1/2）。

【运行前提】
  - 需要 GPU（论文 Table 8：完整微调需 4×RTX 8000；本缩小版单卡可跑）。
  - 无 GPU 时可观看演示视频：https://www.youtube.com/watch?v=SCxq2ye-R4Y
"""

from llm_recovery.load_llm.load_llm import LoadLLM          # 从 HuggingFace 加载基座 LLM 的工具类
from llm_recovery.fine_tuning.lora import LORA              # LoRA 微调封装（setup + supervised_fine_tuning）
import llm_recovery.constants.constants as constants        # 常量定义（如 DEEPSEEK_14B_QWEN 模型名）
from llm_recovery.fine_tuning.examples_dataset import ExamplesDataset  # 指令-答案对 -> 训练 Dataset 的封装
from transformers import set_seed                           # 固定随机种子，保证结果可复现
from datasets import load_dataset

if __name__ == '__main__':
    # 固定全局随机种子（模型初始化、数据打乱、采样等），保证实验可复现。
    seed = 99125
    set_seed(seed)
    # 模型与权重自动分布到可用设备上（多卡自动切分；论文用 4×RTX 8000）。
    device_map = "auto"
    # 加载基座模型：DeepSeek-R1-Distill-Qwen-14B（论文 Remark 1 的 14B 轻量模型）。
    tokenizer, llm = LoadLLM.load_llm(llm_name=constants.LLM.DEEPSEEK_14B_QWEN, device_map=device_map)
    # 加载 68,000 条事故指令数据集（首次运行自动从 HuggingFace 下载约 536MB）。
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="examples_16_june.json")
    # 缩小版实验（论文 E4）：仅取前 5 条指令-答案对，几分钟内可跑完；
    # 完整复现论文结果时去掉 [0:5] 切片即可（训练时长将达数天）。
    instructions = dataset["train"]["instructions"][0][0:5]
    answers = dataset["train"]["answers"][0][0:5]
    # ---- LoRA 超参数（与论文 Table 7 一致）----
    lora_rank = 64        # 秩 r：低秩适配矩阵的秩，越大表达能力越强、显存越多
    lora_alpha = 128      # 缩放系数 α（论文取 2r）
    lora_dropout = 0.05   # 适配层 dropout，防过拟合
    # 给模型注入 LoRA 适配层并准备训练（冻结基座 14B 参数，只训练适配矩阵）。
    llm = LORA.setup_llm_for_fine_tuning(llm=llm, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    # 把指令-答案对封装成训练 Dataset（内部完成分词、构造 labels、
    # 按论文式 (2) 的 teacher-forcing 形式组织样本）。
    dataset = ExamplesDataset(instructions=instructions, answers=answers, tokenizer=tokenizer)
    # ---- 训练超参数（与论文 Table 7 一致的部分已注明）----
    lr = 0.00095                          # 学习率（Table 7；Fig.5 显示高学习率收敛更好）
    per_device_batch_size = 1             # 每卡批大小（Table 7 为 5，此处缩小版取 1）
    num_train_epochs = 1                  # 训练轮数（Table 7 为 2，此处缩小版取 1）
    prompt_logging_frequency = 50         # 每隔多少步打印一次提示/生成样例（便于人工检查）
    max_generation_tokens = 6000          # 训练中日志生成的最大 token 数
    logging_steps = 1                     # 每步记录一次训练指标（Step/Epoch/Loss，见 E4 期望输出）
    running_average_window = 100          # 平滑损失曲线的滑动窗口
    temperature = 0.6                     # 训练日志中生成示例的采样温度（Table 7：0.6）
    save_steps = 25                       # 每 25 步保存一次检查点
    save_limit = 3                        # 最多保留 3 个检查点（节省磁盘）
    gradient_accumulation_steps = 1       # 梯度累积步数（Table 7 为 16，此处缩小版取 1）
    progress_save_frequency = 10          # 每 10 步保存一次训练进度
    # 启动监督微调：内部按论文式 (2) 的交叉熵损失做梯度下降；
    # 运行输出形如 "Step: 1, Epoch: 0.5000, ..."（论文 E4 的期望输出）。
    LORA.supervised_fine_tuning(llm=llm, dataset=dataset, learning_rate=lr,
                                per_device_train_batch_size=per_device_batch_size,
                                num_train_epochs=num_train_epochs, logging_steps=logging_steps, prompts=instructions,
                                answers=answers,
                                prompt_logging=True,
                                running_average_window=running_average_window,
                                max_generation_tokens=max_generation_tokens,
                                prompt_logging_frequency=prompt_logging_frequency, temperature=temperature,
                                save_steps=save_steps, save_limit=save_limit,
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                progress_save_frequency=progress_save_frequency, seed=seed)
