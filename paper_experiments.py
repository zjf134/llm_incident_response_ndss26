# -*- coding: utf-8 -*-
"""
paper_experiments.py —— 多事故实验矩阵 + N 探针（复现论文 Fig 9-12 的数据）
================================================================================
目标：在服务器上生成与论文第 9-11 页实验图对应的数据：
  Fig 9  （p9） 方法 vs 其他模型：恢复时间 / 无效动作率 / 失败率（多事故）
  Fig 10 （p9） 规划每步耗时随候选数 N 的变化（探针）
  Fig 11 （p9） 经验幻觉概率随 N 的变化（探针，逐候选统计）
  Fig 12 （p10）消融：去掉微调 / 规划 / RAG 各自的恢复时间影响
  Fig 13 （p11）与 PPO 对比——需 CSLE 强化学习环境，本脚本不覆盖（报告注明）

与论文的规模差异（报告需注明）：论文 25 事故 × 5 种子；本实验 5 事故 × 1 种子
（每事故固定种子，跨配置可比）+ N 探针 1 事故。其余口径（T=无效动作计 2、
终态=六维全 true）与 paper_pipeline.py 完全一致。

输出（~/llm_ir/paper_figs/）：
  episodes.jsonl     每（模型×配置×事故）一行
  steps.jsonl        每时间步一行（含规划耗时）
  probe.jsonl        探针：每（N×步）的规划耗时与逐候选幻觉标记

运行：conda activate llm_ir && python paper_experiments.py [--dry] [--skip-matrix] [--skip-probe]
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
import time
import argparse
import pathlib
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import paper_pipeline as pp          # 复用 RAG 与算法 1 的全部原语（单一实现源）

OUT_DIR = pathlib.Path.home() / "llm_ir" / "paper_figs"

# 5 个不同事故（取自 examples_16_june.json，按事故描述关键词定位第一条指令）
INCIDENTS = [
    ("FIN-SQL-DB",        "金融 SQL 服务器多阶段攻击"),
    ("Roundcube",         "WEBMAIL01 Roundcube 漏洞利用"),
    ("WordPress",         "WordPress 站点攻陷"),
    ("HMI-STATION-03",    "工业 HMI 站点入侵"),
    ("NAT device",        "边界 NAT 误配置（偏误报类）"),
]

# 实验矩阵：模型 -> [(配置名, planning, rag), ...]
MATRIX = [
    ("ft",     [("rag+plan", "on",  "kb"),
                ("plan",     "on",  "off"),
                ("noplan",   "off", "off")]),
    ("base",   [("plan",     "on",  "off"),
                ("noplan",   "off", "off")]),
    ("qwen05", [("noplan",   "off", "off")]),
]

SEED_BASE = 4200          # 每事故 seed = SEED_BASE + incident_idx（跨配置一致）
# 注意：每个模型必须独立进程运行（--models 过滤 + 多次调用）；
# 同进程内先后加载两个 4-bit 模型会因 bitsandbytes 内存未完全释放而触发
# "modules are dispatched on the CPU" 报错（实测教训）。


def load_model(name):
    print(">> 加载模型 %s ..." % pp.MODELS[name], flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(pp.MODELS[name])
    if name == "qwen05":
        model = AutoModelForCausalLM.from_pretrained(pp.MODELS[name],
                                                     device_map="auto",
                                                     torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(pp.MODELS[name],
                                                     quantization_config=bnb,
                                                     device_map="auto")
    model.eval()
    return model, tok


def resolve_incidents(kb_entries):
    """在数据集中按关键词定位每个事故的第一条指令索引。"""
    from datasets import load_dataset
    ds = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="examples_16_june.json")
    instructions = ds["train"]["instructions"][0]
    picked = []
    for kw, label in INCIDENTS:
        idx = None
        # 第一轮：要求 Incident 段非空且含关键词（保证事故描述完整）
        for i, x in enumerate(instructions):
            seg_i = pp.parse_sections(x).get("incident", "")
            if kw.lower() in seg_i.lower() and len(seg_i) > 40:
                idx = i
                break
        # 回退：全文匹配
        if idx is None:
            for i, x in enumerate(instructions):
                if kw.lower() in x.lower():
                    idx = i
                    break
        if idx is None:
            print("!! 未找到事故关键词 %s，跳过" % kw, flush=True)
            continue
        sec = pp.parse_sections(instructions[idx])
        picked.append((idx, label, sec))
        print("   事故 [%s] -> 指令 #%d | %s" % (label, idx, sec.get("incident", "")[:90]), flush=True)
    return instructions, picked


def run_episode_rich(model, tok, sec, kb_entries, planning, rag, N, M, depth,
                     max_steps, seed, meta):
    """带完整日志的回合执行（主矩阵用；算法与 paper_pipeline.run_episode 一致）。"""
    torch.manual_seed(seed)
    intel, iocs = pp.rag_enrich(json.dumps(sec, ensure_ascii=False), kb_entries, rag != "off")
    state = tuple([False] * 6)
    prev_actions, step_rows = [], []
    T, ineffective, done = 0, 0, False
    t_ep = time.time()
    for t in range(max_steps):
        if all(state):
            done = True
            break
        t_plan = time.time()
        cand_meta = []
        if planning == "on":
            candidates, seen = [], set()
            for _ in range(N * 3):
                a = pp.sample_action(model, tok, sec, intel, state, prev_actions)
                if a and a not in seen:
                    candidates.append(a)
                    seen.add(a)
                if len(candidates) >= N:
                    break
            if not candidates:
                break
            Qs = [statistics.mean(
                    pp.recovery_time(model, tok, sec, intel, state, a, prev_actions, depth)
                    for _ in range(M))
                  for a in candidates]
            action = candidates[Qs.index(min(Qs))]
            cand_meta = [{"action": a[:150], "q": q} for a, q in zip(candidates, Qs)]
        else:
            action = pp.sample_action(model, tok, sec, intel, state, prev_actions)
            if action is None:
                break
            Qs = None
        plan_s = time.time() - t_plan
        s_next, changed = pp.predict_state(model, tok, sec, intel, state, prev_actions, action)
        cost = 1 if changed else 2
        if not changed:
            ineffective += 1
        T += cost
        step_rows.append(dict(meta, t=t, planning_seconds=round(plan_s, 1),
                              n_candidates=(len(cand_meta) if planning == "on" else 1),
                              candidates=cand_meta, chosen=action[:150], cost=cost,
                              changed=changed, state=list(state),
                              next_state=list(s_next)))
        print("  步%d 代价%d 状态%s | %s" % (t, cost, tuple(s_next), action[:80]), flush=True)
        state = s_next
        prev_actions.append(action)
    if all(state):
        done = True
    episode = dict(meta, T=T, terminal=done, actions=len(step_rows),
                   ineffective=ineffective, wall_seconds=round(time.time() - t_ep, 1),
                   intel_used=bool(intel), seed=seed,
                   final_state=list(state))
    return episode, step_rows


def run_probe(model, tok, sec, kb_entries, incident_idx, ns=(1, 2, 3), max_steps=2):
    """N 探针：规划耗时 + 逐候选幻觉统计（候选的模拟状态无变化 = 幻觉，论文定义 4）。"""
    torch.manual_seed(SEED_BASE + incident_idx)
    intel, _ = pp.rag_enrich(json.dumps(sec, ensure_ascii=False), kb_entries, True)
    rows = []
    for N in ns:
        state = tuple([False] * 6)
        prev_actions = []
        for t in range(max_steps):
            if all(state):
                break
            t0 = time.time()
            candidates, seen = [], set()
            for _ in range(N * 3):
                a = pp.sample_action(model, tok, sec, intel, state, prev_actions)
                if a and a not in seen:
                    candidates.append(a)
                    seen.add(a)
                if len(candidates) >= N:
                    break
            cand_stats = []
            for a in candidates:
                # 1 次预测得到模拟状态（兼作幻觉标记），深层递归复用 recovery_time
                s1, changed = pp.predict_state(model, tok, sec, intel, state, prev_actions, a)
                if all(s1):
                    q = 1
                else:
                    a2 = pp.sample_action(model, tok, sec, intel, s1, prev_actions + [a])
                    q = 1 + (1 if a2 is None else
                             pp.recovery_time(model, tok, sec, intel, s1, a2,
                                              prev_actions + [a], 1))
                cand_stats.append({"hallucinating": (not changed), "q": q,
                                   "action": a[:120]})
            plan_s = time.time() - t0
            chosen = min(cand_stats, key=lambda c: c["q"])
            action = chosen["action"]
            s_next, changed = pp.predict_state(model, tok, sec, intel, state, prev_actions, action)
            rows.append({"incident": incident_idx, "N": N, "t": t,
                         "planning_seconds": round(plan_s, 1),
                         "n_candidates": len(cand_stats),
                         "hallu_rate_in_step": round(
                             sum(c["hallucinating"] for c in cand_stats) /
                             max(1, len(cand_stats)), 3),
                         "candidates": cand_stats})
            state = s_next
            prev_actions.append(action)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ft,base,qwen05",
                    help="要运行的模型（逗号分隔），用于分进程续跑")
    ap.add_argument("--append", action="store_true", help="追加写入而非覆盖（续跑用）")
    ap.add_argument("--dry", action="store_true", help="只定位事故，不加载模型")
    ap.add_argument("--skip-matrix", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=4)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kb_path = pathlib.Path(pp.__file__).resolve().parent.parent / "data" / "cve_kb.json"
    if not kb_path.exists():
        kb_path = pathlib.Path.home() / "llm_ir" / "data" / "cve_kb.json"
    kb_entries = json.loads(kb_path.read_text(encoding="utf-8"))
    print("情报库: %d 条" % len(kb_entries), flush=True)

    instructions, picked = resolve_incidents(kb_entries)
    if args.dry:
        print("DRY_OK: %d 个事故就绪" % len(picked), flush=True)
        return

    fmode = "a" if args.append else "w"
    ep_file = open(OUT_DIR / "episodes.jsonl", fmode, encoding="utf-8")
    st_file = open(OUT_DIR / "steps.jsonl", fmode, encoding="utf-8")
    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    if not args.skip_matrix:
        for model_name, configs in MATRIX:
            if model_name not in selected:
                continue
            model, tok = load_model(model_name)
            for cfg_name, planning, rag in configs:
                for idx, label, sec in picked:
                    meta = {"model": model_name, "config": cfg_name, "incident": idx,
                            "incident_label": label, "planning": planning, "rag": rag,
                            "N": args.N, "M": args.M, "depth": args.depth,
                            "max_steps": args.max_steps}
                    print("\n==== %s | %s | %s ====" % (model_name, cfg_name, label), flush=True)
                    ep, steps = run_episode_rich(model, tok, sec, kb_entries, planning, rag,
                                                 args.N, args.M, args.depth, args.max_steps,
                                                 SEED_BASE + idx, meta)
                    ep_file.write(json.dumps(ep, ensure_ascii=False) + "\n")
                    ep_file.flush()
                    for s in steps:
                        st_file.write(json.dumps(s, ensure_ascii=False) + "\n")
                    st_file.flush()
                    print("   => T=%d terminal=%s 无效=%d/%d 用时%.0fs"
                          % (ep["T"], ep["terminal"], ep["ineffective"], ep["actions"],
                             ep["wall_seconds"]), flush=True)
            del model
            torch.cuda.empty_cache()

    if not args.skip_probe:
        model, tok = load_model("ft")
        idx, label, sec = picked[0]
        print("\n==== N 探针（%s，N=1/2/3，2 步）====" % label, flush=True)
        rows = run_probe(model, tok, sec, kb_entries, idx)
        with open(OUT_DIR / "probe.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("PROBE_DONE: %d 行" % len(rows), flush=True)

    ep_file.close()
    st_file.close()
    print("ALL_EXPERIMENTS_DONE", flush=True)


if __name__ == "__main__":
    main()
