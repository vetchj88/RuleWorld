# scripts/baseline_bestofn.py
"""
Phase 1B: Best-of-N Baseline

Performs inference-time search (Best-of-N) on all tasks in the manifest.
For each input, generates N candidate outputs, scores each by mean
completion log-probability, then selects the best using majority vote
or the highest-scoring candidate.

Requirements:
    pip install torch transformers

Usage:
    python baseline_bestofn.py \\
      --manifest ruleworld_manifest_v1.json \\
      --data_root data/ruleworld_v1 \\
      --model Qwen/Qwen2.5-1.5B-Instruct \\
      --n_samples 8 --temperature 0.7 --select maj_logprob
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

# Ensure the root directory is in sys.path so ruleworld package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
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


def safe_name(s: str) -> str:
    return s.replace("/", "-").replace(":", "-")


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
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
    yt, yp = split_tokens(y_true), split_tokens(y_pred)
    if not yt and not yp:
        return 1.0
    return max(0.0, min(1.0, 1.0 - levenshtein(yt, yp) / max(len(yt), len(yp), 1)))


def parse_output_text(full_text: str) -> str:
    """Extract and normalize the completion after the last 'OUTPUT:' marker."""
    line = full_text.split("OUTPUT:")[-1].strip()
    return " ".join(line.split())


def filter_to_allowed_tokens(s: str, allowed: set) -> Tuple[str, bool]:
    toks = split_tokens(s)
    ok = all(t in allowed for t in toks) if toks else True
    return (" ".join(t for t in toks if t in allowed), ok)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def select_candidate(cands: List[str], scores: List[float], mode: str) -> str:
    """
    Select the best candidate from a list using the specified strategy.

    Modes:
        - "best_logprob"  : pick the candidate with the highest mean log-probability.
        - "majority"      : pick the most frequent candidate; break ties arbitrarily.
        - "maj_logprob"   : pick the most frequent candidate; break ties by log-probability.
    """
    if mode in ("majority", "maj_logprob"):
        counts = Counter(cands)
        best_freq = max(counts.values())
        tied = [c for c, k in counts.items() if k == best_freq]
        if len(tied) == 1:
            return tied[0]
        # Break ties by score
        best, best_s = tied[0], float("-inf")
        for c, s in zip(cands, scores):
            if c in tied and s > best_s:
                best, best_s = c, s
        return best
    # Default: highest log-probability
    return cands[max(range(len(scores)), key=lambda i: scores[i])]


# -----------------------
# Generation & scoring
# -----------------------

@torch.inference_mode()
def generate_bestofn_batch(
    model,
    tok,
    prompts: List[str],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """
    Generate n_samples completions for each prompt.

    Returns a flat list of length len(prompts) * n_samples, ordered as:
        [prompt_0_sample_0, prompt_0_sample_1, ..., prompt_1_sample_0, ...]
    """
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n_samples,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    return tok.batch_decode(out, skip_special_tokens=True)


@torch.inference_mode()
def mean_logprob_pairs(
    model,
    tok,
    prompts: List[str],
    completions: List[str],
    batch_size: int,
) -> List[float]:
    """
    Compute mean per-token log-probability of each completion given its prompt.

    Returns a list of floats (one per prompt/completion pair). Lower is better
    in magnitude, so higher (less negative) = more likely completion.
    """
    full_texts = [
        p + (" " if not p.endswith(" ") else "") + c
        for p, c in zip(prompts, completions)
    ]
    scores = []

    for i in range(0, len(full_texts), batch_size):
        batch_texts = full_texts[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        # Prompt lengths (without padding, without special tokens)
        prompt_lens = [
            len(tok(p, add_special_tokens=False)["input_ids"])
            for p in batch_prompts
        ]

        enc = tok(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        logits = model(**enc).logits[:, :-1, :]   # (B, L-1, V)
        labels = enc.input_ids[:, 1:]              # (B, L-1)

        logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        for j, prompt_len in enumerate(prompt_lens):
            # In a left-padded batch, find where the actual (non-pad) tokens start
            full_seq_len = int(enc.attention_mask[j].sum().item())
            seq_len = enc.input_ids.shape[1]
            pad_prefix = seq_len - full_seq_len

            # The completion starts after the prompt tokens (offset by pad prefix)
            # Use index into the *shifted* logp tensor (which has length seq_len - 1)
            comp_start = pad_prefix + prompt_len  # first completion token position in logp
            comp_end   = seq_len - 1              # last valid logp position (= seq_len - 1)

            if comp_end <= comp_start:
                scores.append(float("-inf"))
                continue

            comp_logp = logp[j, comp_start:comp_end]
            scores.append(comp_logp.mean().item())

    return scores


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Best-of-N inference-time search baseline for RuleWorld."
    )
    ap.add_argument("--manifest", required=True, help="Path to ruleworld_manifest_v1.json")
    ap.add_argument("--data_root", required=True, help="Root dataset directory")
    ap.add_argument("--model", required=True, help="HuggingFace model name or local path")
    ap.add_argument("--split", default="eval", help="Dataset split to evaluate on")
    ap.add_argument("--n", type=int, default=50, help="Number of examples per task")
    ap.add_argument("--shots", type=int, default=4, help="Number of in-context few-shot examples")
    ap.add_argument("--n_samples", type=int, default=8, help="Number of candidates to generate per input (N in Best-of-N)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--select", default="maj_logprob",
                    choices=["best_logprob", "majority", "maj_logprob"],
                    help="Candidate selection strategy.")
    ap.add_argument("--batch_size", type=int, default=2,
                    help="Number of prompts to generate for simultaneously.")
    ap.add_argument("--score_batch_size", type=int, default=4,
                    help="Batch size for log-probability scoring.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--strict_vocab_filter", action="store_true",
                    help="Filter outputs to allowed RuleWorld vocabulary before scoring.")
    ap.add_argument("--run_dir", default="",
                    help="Explicit output directory. Auto-generated if not set.")
    args = ap.parse_args()

    set_seed(args.seed)

    manifest = load_manifest(args.manifest)
    allowed = allowed_token_set(manifest)
    tasks = sorted(manifest["tasks"], key=lambda t: int(t["index"]))

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = os.path.join(
            "runs",
            f"bestofN_{safe_name(args.model)}_N{args.n_samples}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    ensure_dir(run_dir)

    table_path = os.path.join(run_dir, "bestofN_table.csv")
    fieldnames = ["task", "split", "n", "shots", "N", "temp", "select",
                  "mean_reward", "exact_rate", "wall_time_sec"]
    with open(table_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # Left-padding is required for correct batched generation
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16
    ).eval()

    for task in tasks:
        tid = task["task_id"]
        rows = read_jsonl(os.path.join(args.data_root, tid, f"{args.split}.jsonl"))[: args.n]
        if not rows:
            continue

        print(f"Eval {tid} ({len(rows)} examples)...", flush=True)
        prompts = [build_fewshot_prompt(args.data_root, tid, r["x"], args.shots) for r in rows]
        y_trues = [r["y"] for r in rows]

        rewards, exact = [], 0
        t0 = time.time()

        for i in range(0, len(rows), args.batch_size):
            mb_p = prompts[i : i + args.batch_size]
            mb_y = y_trues[i : i + args.batch_size]

            # Generate N candidates per prompt (flat list, grouped by prompt)
            cands_flat = generate_bestofn_batch(
                model, tok, mb_p,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            # Parse completions out of each decoded string
            cands_flat = [parse_output_text(c) for c in cands_flat]

            # Score: expand prompts to match the flat candidate list
            prompts_expanded = [p for p in mb_p for _ in range(args.n_samples)]
            scores_flat = mean_logprob_pairs(
                model, tok, prompts_expanded, cands_flat, args.score_batch_size
            )

            for j in range(len(mb_p)):
                cands  = cands_flat [j * args.n_samples : (j + 1) * args.n_samples]
                scores = scores_flat[j * args.n_samples : (j + 1) * args.n_samples]

                if args.strict_vocab_filter:
                    cands = [filter_to_allowed_tokens(c, allowed)[0] for c in cands]

                chosen = select_candidate(cands, scores, args.select)
                rewards.append(token_edit_reward(mb_y[j], chosen))
                exact += int(chosen == mb_y[j])

        wall = time.time() - t0
        mean_r = sum(rewards) / len(rewards)

        with open(table_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                "task": tid, "split": args.split, "n": len(rewards),
                "shots": args.shots, "N": args.n_samples,
                "temp": args.temperature, "select": args.select,
                "mean_reward": round(mean_r, 6),
                "exact_rate": round(exact / len(rewards), 6),
                "wall_time_sec": round(wall, 4),
            })
        print(f"  -> Reward: {mean_r:.4f}", flush=True)

    print("Best-of-N Eval Complete.")


if __name__ == "__main__":
    main()