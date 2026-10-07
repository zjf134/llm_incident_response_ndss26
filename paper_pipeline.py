# -*- coding: utf-8 -*-
"""
paper_pipeline.py —— 论文完整管线的源码集成：RAG（§IV-C）+ 前瞻规划（§IV-D 算法 1）
================================================================================
管线（对应论文 Fig. 2b）：
    系统日志 --[RAG: IOC 提取 -> 威胁情报检索 -> 拼入提示]--> 微调 LLM
    --[规划: 采样 N 候选 -> 每候选模拟 M 条恢复轨迹 -> argmin Q̃]--> 选定动作
    --[状态更新 p(s'|s,a,I)]--> 循环直至终态 (1,1,1,1,1,1)

模型选择：
    --model ft      论文微调模型 kimhammar/LLMIncidentResponse（4-bit，双卡）
    --model base    微调前基座 deepseek-ai/DeepSeek-R1-Distill-Qwen-14B（消融：无微调）
    --model qwen05  Qwen2.5-0.5B-Instruct（其他模型对照，fp16）

指标（论文 §VII 口径）：
    恢复时间 T：每动作 1 时间单位，无效动作（未改变状态，定义 4）计 2；
    到达 max-steps 未终态记为"未恢复"（论文的 % failed recoveries）。

用法示例（服务器）：
    python paper_pipeline.py --model ft --planning on --rag kb --incident 0
    python paper_pipeline.py --model ft --planning off --incident 0
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
import argparse
import pathlib
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = {
    "ft": "kimhammar/LLMIncidentResponse",
    "base": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
}
FIELDS = ["is_attack_contained", "is_knowledge_sufficient", "are_forensics_preserved",
          "is_eradicated", "is_hardened", "is_recovered"]

STATE_PRED_INSTR = (
    "You are a security operator with advanced knowledge in cybersecurity and IT systems. "
    "Based on the system description, the logs, the incident, the current state and the "
    "previous recovery actions, determine the next state of the recovery process. "
    "It is important that the state only changes if the action is effective in achieving one "
    "of the recovery goals: containment, information gathering, preserving evidence, "
    "eradication, hardening, or recovery. A state variable can only change from 'false' to "
    "'true', it cannot be changed from 'true' to 'false'. Return a JSON object that defines "
    "the next state and contains the Boolean fields 'is_attack_contained', "
    "'is_knowledge_sufficient', 'are_forensics_preserved', 'is_eradicated', 'is_hardened', "
    "'is_recovered'."
)
ACTION_GEN_INSTR = (
    "You are a security operator with advanced knowledge in cybersecurity and IT systems. "
    "You have been given information about a security incident and should generate the next "
    "suitable action for recovering the system from the incident. Your suggested action "
    "should be based on the logs, the system description, the current state, and the previous "
    "recovery actions only. Make sure that the suggested recovery action is consistent with "
    "the system description and the logs and that you do not repeat any action that has "
    "already been performed. When selecting the recovery action, make sure that it is "
    "concrete and actionable and minimizes unnecessary service disruptions. Return a JSON "
    "object with two properties: 'Action' and 'Explanation', both of which should be strings."
)


# ───────────────────── RAG（论文 §IV-C，Fig. 7）─────────────────────
IOC_PATTERNS = [(r"CVE-\d{4}-\d{4,7}", "cve"),
                (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "ipv4")]


def extract_iocs(text):
    found = []
    for pat, typ in IOC_PATTERNS:
        for m in re.findall(pat, text):
            if m not in found:
                found.append(m)
    return found


def kb_retrieve(text, kb_entries, top_k=2):
    """威胁情报检索：IOC 与 ioc_hints 关键词在日志/事故文本中的命中数打分。
    （论文用 OTX API 在线检索；离线复现用本地情报库，检索思想相同。）"""
    t = text.lower()
    scored = []
    for e in kb_entries:
        hits = [h for h in e.get("ioc_hints", []) if h.lower() in t]
        if hits:
            scored.append((len(hits), e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:top_k]]


def format_intel(entries):
    lines = []
    for e in entries:
        lines.append("- %s (%s, severity: %s): %s Countermeasures: %s"
                     % (e["name"], e["key"], e["severity"], e["summary"],
                        "; ".join(e["mitigations"][:2])))
    return "\n".join(lines)


def rag_enrich(text, kb_entries, use_rag):
    """论文 Fig.7 第 1-2 步：提取 IOC -> 检索情报 -> 返回情报块（无命中返回空）。"""
    if not use_rag:
        return "", []
    iocs = extract_iocs(text)
    entries = kb_retrieve(text, kb_entries)
    if not entries:
        return "", iocs
    return format_intel(entries), iocs


# ──────────────── 提示词构造（论文附录 B 模板）────────────────────
def parse_sections(instruction):
    parts = re.split(r"^###\s*", instruction, flags=re.M)
    out = {}
    for p in parts:
        m = re.match(r"(\w[\w ]*):\s*(.*)", p, flags=re.S)
        if m:
            out[m.group(1).strip().lower()] = m.group(2).strip()
    return out


def build_prompt(sec, intel, state, prev_actions, instr):
    prev = "\n".join("%d. %s" % (i + 1, a) for i, a in enumerate(prev_actions)) or "None"
    state_txt = "; ".join("%s: %s" % (f, "true" if s else "false")
                          for f, s in zip(FIELDS, state))
    intel_block = ("\n### Threat intelligence (retrieved):\n%s\n" % intel) if intel else ""
    return ("### System:\n%s\n### Logs:\n%s\n%s### Incident:\n%s\n"
            "### State:\n%s\n### Previous recovery actions:\n%s\n"
            "### Instruction:\n%s\n### Response:"
            % (sec.get("system", ""), sec.get("logs", ""), intel_block,
               sec.get("incident", ""), state_txt, prev, instr))


def extract_json(text):
    m = re.search(r"\{[^{}]*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=400, temperature=0.6):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         temperature=temperature, do_sample=True,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    tail = text.split("</think>")[-1]
    return text, (extract_json(tail) or extract_json(text))


# ──────────────── 算法 1 组件（候选 / 状态预测 / Q̃）────────────────
def sample_action(model, tok, sec, intel, state, prev_actions):
    prompt = build_prompt(sec, intel, state, prev_actions, ACTION_GEN_INSTR)
    text, j = generate(model, tok, prompt, max_new_tokens=500)
    if j and isinstance(j.get("Action"), str) and j["Action"].strip():
        return j["Action"].strip()
    return None


def predict_state(model, tok, sec, intel, state, prev_actions, action):
    prompt = build_prompt(sec, intel, state, prev_actions + [action], STATE_PRED_INSTR)
    text, j = generate(model, tok, prompt, max_new_tokens=400)
    if j is None:
        return state, False                     # 预测失败 = 无效动作（定义 4）
    nxt, changed = list(state), False
    for k, f in enumerate(FIELDS):
        if f in j and bool(j[f]) and not nxt[k]:
            nxt[k] = True
            changed = True
    return tuple(nxt), changed


def recovery_time(model, tok, sec, intel, state, action, prev_actions, depth):
    """算法 1 第 13-22 行 RECOVERY-TIME；c(s,a)=1（论文 §V 的证明设定）。"""
    s_next, _ = predict_state(model, tok, sec, intel, state, prev_actions, action)
    cost = 1
    if all(s_next) or depth <= 0:
        return cost
    a_next = sample_action(model, tok, sec, intel, s_next, prev_actions + [action])
    if a_next is None:
        return cost + 1
    return cost + recovery_time(model, tok, sec, intel, s_next, a_next,
                                prev_actions + [action], depth - 1)


# ──────────────────────── 回合执行 ────────────────────────
def run_episode(model, tok, sec, kb_entries, cfg):
    intel, iocs = rag_enrich(json.dumps(sec, ensure_ascii=False), kb_entries, cfg["rag"] != "off")
    if cfg["rag"] == "otx":                     # OTX 在线情报（需 OTX_API_KEY）
        try:
            from otx_client import enrich_with_otx
            otx_info, _ = enrich_with_otx({i: "cve" if i.startswith("CVE") else "ipv4"
                                           for i in iocs})
            if otx_info:
                intel = (intel + "\n" if intel else "") + otx_info
        except Exception:
            pass
    state = tuple([False] * 6)
    prev_actions, steps_trace = [], []
    T, ineffective, done = 0, 0, False
    for t in range(cfg["max_steps"]):
        if all(state):
            done = True
            break
        if cfg["planning"] == "on":
            # 算法 1：采样 N 候选 -> 每候选 M 条轨迹估 Q̃ -> argmin
            candidates, seen = [], set()
            for _ in range(cfg["N"] * 3):
                a = sample_action(model, tok, sec, intel, state, prev_actions)
                if a and a not in seen:
                    candidates.append(a)
                    seen.add(a)
                if len(candidates) >= cfg["N"]:
                    break
            if not candidates:
                break
            Qs = [statistics.mean(
                    recovery_time(model, tok, sec, intel, state, a, prev_actions, cfg["depth"])
                    for _ in range(cfg["M"]))
                  for a in candidates]
            action = candidates[Qs.index(min(Qs))]
        else:
            action = sample_action(model, tok, sec, intel, state, prev_actions)
            if action is None:
                break
        s_next, changed = predict_state(model, tok, sec, intel, state, prev_actions, action)
        step_cost = 1 if changed else 2         # 无效动作计 2（论文 §VII 口径）
        if not changed:
            ineffective += 1
        T += step_cost
        steps_trace.append({"t": t, "state": list(state), "action": action,
                            "Q": (Qs if cfg["planning"] == "on" else None),
                            "next_state": list(s_next), "changed": changed,
                            "cost": step_cost})
        print("  步 %d | 代价 %d | 状态 %s | %s" % (t, step_cost, tuple(s_next), action[:90]),
              flush=True)
        state = s_next
        prev_actions.append(action)
    if all(state):                              # 循环耗尽后补终态判定
        done = True
    return {"T": T, "terminal": done, "ineffective": ineffective,
            "actions": len(steps_trace), "steps": steps_trace, "intel_used": bool(intel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="ft")
    ap.add_argument("--planning", choices=["on", "off"], default="on")
    ap.add_argument("--rag", choices=["off", "kb", "otx"], default="off")
    ap.add_argument("--incident", type=int, default=0)
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--out", default=None, help="结果 JSONL 追加文件")
    args = ap.parse_args()

    print("[1/3] 加载模型 %s（4-bit，双卡）..." % MODELS[args.model], flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    if args.model == "qwen05":
        model = AutoModelForCausalLM.from_pretrained(MODELS[args.model],
                                                     device_map="auto",
                                                     torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODELS[args.model],
                                                     quantization_config=bnb,
                                                     device_map="auto")
    model.eval()

    print("[2/3] 加载数据集与威胁情报库...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="examples_16_june.json")
    instruction = ds["train"]["instructions"][0][args.incident]
    sec = parse_sections(instruction)
    kb_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "cve_kb.json"
    if not kb_path.exists():                     # 兼容仓库外运行（服务器）
        kb_path = pathlib.Path.home() / "llm_ir" / "data" / "cve_kb.json"
    kb_entries = json.loads(kb_path.read_text(encoding="utf-8"))

    print("[3/3] 运行回合（planning=%s, rag=%s）..." % (args.planning, args.rag), flush=True)
    cfg = vars(args)
    result = run_episode(model, tok, sec, kb_entries, cfg)
    result.update({"model": args.model, "planning": args.planning, "rag": args.rag,
                   "incident": args.incident, "N": args.N, "M": args.M,
                   "depth": args.depth, "max_steps": args.max_steps})
    print(json.dumps({k: v for k, v in result.items() if k != "steps"},
                     ensure_ascii=False), flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print("EPISODE_DONE", flush=True)


if __name__ == "__main__":
    main()
