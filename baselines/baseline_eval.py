# scripts/baseline_eval.py
"""
Phase 1A: Frozen Baseline (Batched)

Evaluates a frozen (untrained) model — zero-shot or few-shot — on all tasks
in the manifest. Produces a CSV table of per-task reward, exact-match rate,
and parse validity rate.

Requirements:
    pip install torch transformers

Usage:
    python baseline_eval.py \\
      --manifest ruleworld_manifest_v1.json \\
      --data_root data/ruleworld_v1 \\
      --model Qwen/Qwen2.5-1.5B-Instruct \\
      --shots 4 --n 100 --temperature 0.0
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure the root directory is in sys.path so ruleworld package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ruleworld.prompting import build_fewshot_prompt, read_jsonl
from ruleworld.utils import levenshtein


# -----------------------
# Utilities
# -----------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_model_name(s: str) -> str:
    return s.replace("/", "-").replace("\\", "-").replace(":", "-")


def try_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def allowed_token_set(manifest: Dict[str, Any]) -> set:
    meta = manifest["meta"]
    return set(
        list(meta.get("vocab_S", []))
        + list(meta.get("vocab_D", []))
        + [meta.get("marker_token", "zz")]
    )


def split_tokens(s: str) -> List[str]:
    return [t for t in s.strip().split() if t]


def token_edit_reward(y_true: str, y_pred: str) -> float:
    yt = split_tokens(y_true)
    yp = split_tokens(y_pred)
    if not yt and not yp:
        return 1.0
    return max(0.0, min(1.0, 1.0 - levenshtein(yt, yp) / max(len(yt), len(yp), 1)))


def parse_output_text(full_text: str) -> str:
    """Extract and normalize the completion following the last 'OUTPUT:' marker."""
    idx = full_text.rfind("OUTPUT:")
    tail = full_text[idx + len("OUTPUT:"):] if idx >= 0 else full_text
    line = tail.splitlines()[0].strip() if tail.splitlines() else tail.strip()
    return " ".join(t for t in line.split() if t)


def filter_to_allowed_tokens(s: str, allowed: set) -> Tuple[str, bool]:
    toks = split_tokens(s)
    ok = all(t in allowed for t in toks) if toks else True
    return (" ".join(t for t in toks if t in allowed), ok)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def batched_idxs(n: int, bs: int) -> Iterable[List[int]]:
    for i in range(0, n, bs):
        yield list(range(i, min(n, i + bs)))


# -----------------------
# Generation
# -----------------------

@torch.inference_mode()
def generate_batch(
    model,
    tok,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """
    Batched model.generate(). Returns decoded strings (prompt + completion) for each item.
    Tokenizer must have padding_side='left' for correct batched generation.
    """
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    if temperature > 0.0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs["do_sample"] = False

    out = model.generate(**enc, **gen_kwargs)
    return tok.batch_decode(out, skip_special_tokens=True)


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Frozen (no-training) few-shot baseline for RuleWorld."
    )
    ap.add_argument("--manifest", type=str, required=True,
                    help="Path to ruleworld_manifest_v1.json")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root dataset directory (e.g. data/ruleworld_v1)")
    ap.add_argument("--model", type=str, required=True,
                    help="HuggingFace model name or local path")
    ap.add_argument("--split", type=str, default="eval",
                    help="Dataset split to evaluate on")
    ap.add_argument("--n", type=int, default=100,
                    help="Maximum number of examples to evaluate per task")
    ap.add_argument("--shots", type=int, default=4,
                    help="Number of in-context few-shot examples")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Decoding temperature (0 = greedy)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=2,
                    help="Inference batch size")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--strict_vocab_filter", action="store_true",
                    help="Filter outputs to the allowed RuleWorld vocabulary before scoring.")
    ap.add_argument("--run_dir", type=str, default="",
                    help="Explicit output directory. Auto-generated under runs/ if not set.")
    ap.add_argument("--write_examples", action="store_true",
                    help="Write per-example predictions to predictions.jsonl.")
    args = ap.parse_args()

    set_seed(args.seed)

    manifest = load_manifest(args.manifest)
    allowed = allowed_token_set(manifest)
    tasks = sorted(manifest["tasks"], key=lambda t: int(t["index"]))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_id = f"baseline_frozen_{safe_model_name(args.model)}_n{args.n}_shots{args.shots}_{stamp}"
        run_dir = os.path.join("runs", run_id)
    ensure_dir(run_dir)

    table_path = os.path.join(run_dir, "baseline_table.csv")
    pred_path  = os.path.join(run_dir, "predictions.jsonl")

    table_fields = [
        "task", "split", "n", "shots", "temp",
        "mean_reward", "exact_rate", "parse_ok_rate",
        "wall_time_sec", "examples_per_sec",
    ]
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=table_fields).writeheader()

    # Left-padding is required for correct batched generation
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16
    ).eval()

    pred_f = open(pred_path, "w", encoding="utf-8") if args.write_examples else None

    for task in tasks:
        task_id = task["task_id"]
        rows = read_jsonl(os.path.join(args.data_root, task_id, f"{args.split}.jsonl"))[: args.n]
        if not rows:
            continue

        print(f"Eval {task_id} ({len(rows)} examples)...", flush=True)
        prompts = [build_fewshot_prompt(args.data_root, task_id, r["x"], args.shots) for r in rows]
        y_trues = [r["y"] for r in rows]

        rewards, exact, parse_ok = [], 0, 0
        t0 = time.time()

        for idxs in batched_idxs(len(prompts), args.batch_size):
            batch_p = [prompts[i] for i in idxs]
            batch_y = [y_trues[i] for i in idxs]

            outputs = generate_batch(model, tok, batch_p, args.max_new_tokens,
                                     args.temperature, args.top_p)

            for full, yt in zip(outputs, batch_y):
                yp = parse_output_text(full)
                if args.strict_vocab_filter:
                    yp, ok = filter_to_allowed_tokens(yp, allowed)
                else:
                    toks = split_tokens(yp)
                    ok = all(t in allowed for t in toks) if toks else True

                parse_ok += int(ok)
                exact += int(yp == yt)
                rwd = token_edit_reward(yt, yp)
                rewards.append(rwd)

                if pred_f:
                    pred_f.write(json.dumps({"task": task_id, "y_true": yt,
                                             "y_pred": yp, "reward": rwd}) + "\n")

        wall = time.time() - t0
        mean_r = sum(rewards) / len(rewards)

        with open(table_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=table_fields).writerow({
                "task": task_id, "split": args.split,
                "n": len(rewards), "shots": args.shots, "temp": args.temperature,
                "mean_reward":    round(mean_r,                       6),
                "exact_rate":     round(exact    / len(rewards),      6),
                "parse_ok_rate":  round(parse_ok / len(rewards),      6),
                "wall_time_sec":  round(wall,                         4),
                "examples_per_sec": round(len(rewards) / wall,        4),
            })
        print(f"  -> Reward: {mean_r:.4f}", flush=True)

    if pred_f:
        pred_f.close()
    print("Baseline Eval Complete.")


if __name__ == "__main__":
    main()
