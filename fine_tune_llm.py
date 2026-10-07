# -*- coding: utf-8 -*-
"""
fine_tune_llm.py —— 步骤 (S4)：在事故响应数据集上微调 LLM
====================================================================

【本脚本在论文中的位置】
  - 对应论文附录 F 的步骤 (S4)（Experiment E4）。
  - 为便于快速验证，只取 5 条样本微调（论文 E4 同为缩小版设计）；
    论文完整微调用全部 68,000 条数据、4×RTX 8000 训练约 800 分钟（论文 Fig. 5）。

【4-bit 适配说明（与原版的三处差异，模型与超参数不变）】
  1. 计算精度 fp16：原版经 LoadLLM.load_llm 使用 bfloat16 计算、训练循环
     bf16=True——图灵架构显卡（2080 Ti/T4）不支持 bf16，改用 fp16；
  2. 样本构造：原版取 instructions[0:5]（长样本），在 11 GB 显存上损失计算
     阶段会 OOM（Qwen 词表 15 万，logits 显存 ∝ 序列长度）——改为
     "答案最短的 5 条 + 指令从左侧截断至 448 token（答案完整保留）"；
  3. 层分配：手工把前 30 层放 GPU0、后 18 层+输出头放 GPU1（同论文库
     LoadLLM.create_device_map 的手法），为反向传播留出余量。
  超参数与论文 Table 7 完全一致：LoRA r=64、alpha=128、dropout=0.05、
  lr=0.00095、batch=1、epochs=1（本脚本缩小版取值）。

【对应的论文核心内容】
  * 监督微调目标 = 答案上的逐 token 交叉熵损失（论文 §IV-B 式 (2)）；
  * LoRA 只训练约 5033 万个低秩适配参数（运行时输出
    "Trainable parameters: 50331648"，相比 14B 全参数微调节省约 300 倍显存）；
  * 数据集含"动作生成"与"状态预测"两类指令，使模型既能生成响应动作，
    又能预测六维恢复状态（定义 1）——后者是规划算法（Alg. 1）模拟轨迹的基础。

【运行前提】
  - 需要 GPU（缩小版单张 11GB 卡即可）；
  - 无 GPU 时观看演示视频：https://www.youtube.com/watch?v=SCxq2ye-R4Y
"""

from llm_recovery.fine_tuning.lora import LORA              # LoRA 微调封装
import llm_recovery.constants.constants as constants        # 常量（基座模型名等）
from llm_recovery.fine_tuning.examples_dataset import ExamplesDataset  # 仅复用其 collate
from llm_recovery.fine_tuning.logging_callback import LoggingCallback
from transformers import (set_seed, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments,
                          PrinterCallback, ProgressCallback)
from datasets import load_dataset
import torch.utils.data as tud

if __name__ == '__main__':
    # 固定全局随机种子，保证实验可复现。
    seed = 99125
    set_seed(seed)

    # ---- [4-bit 适配 1] 直接加载基座模型（不再经 LoadLLM 的 bf16 路径）----
    print("[1/4] 4-bit NF4 加载基座模型（手工双卡映射）...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,          # 双重量化（论文 Table 7）
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,    # 图灵卡不支持 bf16 → fp16
    )
    tokenizer = AutoTokenizer.from_pretrained(constants.LLM.DEEPSEEK_14B_QWEN)
    # 手工设备映射（同论文库 LoadLLM.create_device_map 的手法）：
    # auto 拆分会把 lm_head+交叉熵临时张量压在 GPU1 上导致反向 OOM；
    # 这里 GPU0 扛 embed+前 30 层，GPU1 留 18 层+输出头。
    device_map = {"model.embed_tokens": 0}
    for i in range(30):
        device_map["model.layers.%d" % i] = 0
    for i in range(30, 48):
        device_map["model.layers.%d" % i] = 1
    device_map["model.norm"] = 1
    device_map["model.rotary_emb"] = 1
    device_map["lm_head"] = 1
    llm = AutoModelForCausalLM.from_pretrained(
        constants.LLM.DEEPSEEK_14B_QWEN,         # 基座模型（Remark 1 的 14B）
        quantization_config=bnb,
        device_map=device_map,
    )
    llm.eval()

    # ---- [4-bit 适配 2] 样本构造：答案最短的 5 条 + 指令左截断至 448 token ----
    print("[2/4] 加载数据集，构造 5 条训练样本（答案完整保留，指令保留末尾）...")
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1",
                           data_files="examples_16_june.json")
    instructions = list(dataset["train"]["instructions"][0])
    answers = list(dataset["train"]["answers"][0])
    MAX_LEN = 448
    ans_lens = [len(tokenizer(y)["input_ids"]) for y in answers]
    items, prompt_texts, answer_texts = [], [], []
    for i in sorted(range(len(answers)), key=lambda k: ans_lens[k])[:5]:
        yi = tokenizer(answers[i], add_special_tokens=False)["input_ids"][:MAX_LEN - 80]
        room = MAX_LEN - len(yi)
        xi = tokenizer(instructions[i], add_special_tokens=False)["input_ids"][-room:]
        # 训练目标（论文式 (2)）：labels 中指令段置 -100，只对答案 token 计算交叉熵。
        items.append((
            torch.tensor(xi + yi, dtype=torch.long),
            torch.tensor([-100] * len(xi) + yi, dtype=torch.long),
        ))
        prompt_texts.append(tokenizer.decode(xi))
        answer_texts.append(tokenizer.decode(yi))

    class LeftTruncatedDataset(tud.Dataset):
        """样本已预构造为 (input_ids, labels)；collate 复用 ExamplesDataset 实现。"""

        def __init__(self, items, tokenizer):
            self.items = items
            self.tokenizer = tokenizer
            self._collator = ExamplesDataset(instructions=[], answers=[],
                                             tokenizer=tokenizer)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            input_ids, labels = self.items[idx]
            return {"input_ids": input_ids,
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": labels}

        def collate(self, batch):
            return self._collator.collate(batch)

    ft_dataset = LeftTruncatedDataset(items, tokenizer)

    # ---- LoRA 注入（超参数与论文 Table 7 一致）----
    print("[3/4] 注入 LoRA 适配器（r=64, alpha=128, dropout=0.05）...")
    llm = LORA.setup_llm_for_fine_tuning(llm=llm, r=64, lora_alpha=128, lora_dropout=0.05)

    # ---- [4-bit 适配 3] 训练循环：复刻 LORA.supervised_fine_tuning，仅 fp16 ----
    print("[4/4] 训练（5 条 × 1 轮；fp16 替代 bf16，其余与库实现一致）...")
    lr = 0.00095                          # 学习率（Table 7；Fig.5 高学习率收敛更好）
    temperature = 0.6                     # 训练日志中生成示例的采样温度（Table 7）
    args = TrainingArguments(
        output_dir="ds-dt-lora",
        fp16=True,                        # 原库 bf16=True；图灵卡改 fp16
        per_device_train_batch_size=1,    # 批大小（Table 7 为 5，缩小版取 1）
        num_train_epochs=1,               # 训练轮数（Table 7 为 2，缩小版取 1）
        learning_rate=lr,
        logging_steps=1,                  # 每步记录（输出形如 Step: 1, Epoch: ...）
        save_strategy="steps", save_steps=25, save_total_limit=3,
        gradient_accumulation_steps=1,
        seed=seed, report_to=[],
    )
    callback = LoggingCallback(
        prompts=prompt_texts, answers=answer_texts, tokenizer=tokenizer,
        dataset=ft_dataset, window=100,
        gen_kwargs=dict(max_new_tokens=6000, temperature=temperature, do_sample=True),
        prompt_logging=False, prompt_logging_frequency=50,
        progress_save_frequency=10, seed=seed,
    )
    trainer = Trainer(model=llm, args=args, train_dataset=ft_dataset,
                      data_collator=ft_dataset.collate, callbacks=[callback])
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.train()
    trainer.save_model(output_dir="ds-dt-lora")
    print("E4_DONE: LoRA 适配器已保存到 ./ds-dt-lora")
