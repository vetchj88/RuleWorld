# scripts/baseline_lora_sequential.py
"""
Baseline 3 — Sequential LoRA training (continual learning baseline)

What it does:
- Loads tasks from ruleworld_manifest_v1.json (authoritative order)
- For each task in a stream:
  1) Train LoRA adapters on that task for a fixed budget (steps)
  2) Evaluate on ALL tasks seen so far (and optionally a retention split)
- Logs a full run folder with:
  - config.json
  - train_log.csv        (per train step)
  - eval_matrix.csv      (after each task, eval on tasks 1..t)
  - aggregates.csv       (after each task, average over seen tasks)
  - summary.json         (incl. per-task best/forgetting)

Key baseline properties:
- Single LoRA adapter updated sequentially (no replay, no regularization)
- This is the "strawman" continual learning baseline that stronger systems should beat.

Requirements:
    pip install torch transformers peft

Suggested run (Qwen 1.5B, ~12 GB VRAM):
    python -u baseline_lora_sequential.py \\
      --manifest ruleworld_manifest_v1.json \\
      --data_root data/ruleworld_v1 \\
      --model Qwen/Qwen2.5-1.5B-Instruct \\
      --train_split train --eval_split eval --retention_split retention \\
      --steps_per_task 200 \\
      --train_batch_size 2 --grad_accum 8 \\
      --eval_batch_size 2 \\
      --train_shots 0 --eval_shots 4 \\
      --max_new_tokens 32 \\
      --lr 2e-4 \\
      --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \\
      --require_cuda --strict_vocab_filter

Resume from an existing run:
    python -u baseline_lora_sequential.py --run_dir runs/<existing_run> --resume
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure the root directory is in sys.path so ruleworld package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, TaskType, get_peft_model
from ruleworld.prompting import build_fewshot_prompt, read_jsonl
from ruleworld.utils import levenshtein, sha256_file


# -----------------------
# Utilities
# -----------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    S = list(meta.get("vocab_S", []))
    D = list(meta.get("vocab_D", []))
    marker = meta.get("marker_token", "zz")
    return set(S + D + [marker])


def split_tokens(s: str) -> List[str]:
    return [t for t in s.strip().split(" ") if t]


def token_edit_reward(y_true: str, y_pred: str) -> float:
    yt = split_tokens(y_true)
    yp = split_tokens(y_pred)
    if not yt and not yp:
        return 1.0
    denom = max(len(yt), len(yp), 1)
    return max(0.0, min(1.0, 1.0 - levenshtein(yt, yp) / denom))


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


def build_prompt(data_root: str, task_id: str, x: str, shots: int) -> str:
    if shots <= 0:
        return f"You will transform an INPUT sequence into an OUTPUT sequence.\n\nINPUT: {x}\nOUTPUT:"
    return build_fewshot_prompt(data_root, task_id, x, n_shots=shots)


def detect_lora_target_modules(model: nn.Module) -> List[str]:
    """
    Auto-detect common projection module names for LoRA targeting.
    Returns a sorted list of leaf module name components (as expected by PEFT).
    Falls back to all Linear leaf names if none of the common names are found.
    """
    common = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found: set = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in common:
                found.add(leaf)
    if not found:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                found.add(name.split(".")[-1])
    return sorted(found)


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# -----------------------
# Supervised batch construction
# -----------------------

@dataclass
class TrainBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def make_supervised_batch(
    tok,
    prompts: List[str],
    completions: List[str],
    max_seq_len: int,
) -> TrainBatch:
    """
    Encode full_text = prompt + completion and build labels that mask prompt tokens.

    The key challenge with batched left-padding: the prompt tokens do not start at
    index 0 in the padded full sequence. We must locate where each prompt ends by
    comparing token-level lengths from the un-padded encodings, then offset by the
    leading padding in the padded full encoding.
    """
    full_texts = []
    prompt_texts = []
    for p, c in zip(prompts, completions):
        sep = "" if p.endswith(" ") else " "
        full_texts.append(p + sep + c)
        prompt_texts.append(p + sep)

    # Encode with padding so tensors are uniform shape
    enc_full = tok(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )

    # Encode prompts without padding to get exact prompt token lengths
    enc_prompt_npad = tok(
        prompt_texts,
        return_tensors="pt",
        padding=False,  # no padding — we want actual lengths
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
    )

    input_ids = enc_full["input_ids"]       # (B, L)
    attention_mask = enc_full["attention_mask"]  # (B, L)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # mask all padding first

    for i, prompt_enc in enumerate(enc_prompt_npad["input_ids"]):
        prompt_len = len(prompt_enc)

        # Count leading pad tokens in the full (padded) encoding for this example
        full_len = int(attention_mask[i].sum().item())
        seq_len = input_ids.shape[1]
        pad_prefix = seq_len - full_len  # number of pad tokens prepended

        # Mask out the prompt portion (after the pad prefix)
        mask_end = pad_prefix + prompt_len
        if mask_end > 0:
            labels[i, :mask_end] = -100

    return TrainBatch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


# -----------------------
# Batched generation for eval
# -----------------------

@torch.inference_mode()
def generate_batch(
    model,
    tok,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_max_prompt_len: Optional[int] = None,
) -> List[str]:
    """
    Batched model.generate(). Returns decoded strings (prompt + completion) for each item.
    Tokenizer must have padding_side='left' for correct batched generation.
    """
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=batch_max_prompt_len,
    )
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


def batched(iterable: List[Any], bs: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), bs):
        yield iterable[i : i + bs]


def eval_task_rows(
    model,
    tok,
    data_root: str,
    task_id: str,
    rows: List[Dict[str, Any]],
    shots: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    allowed: set,
    strict_vocab_filter: bool,
    eval_batch_size: int,
) -> Dict[str, float]:
    """Evaluate a list of (x, y) rows for one task. Returns aggregate metrics."""
    model.eval()
    rewards: List[float] = []
    exact = 0
    parse_ok = 0

    prompts = [build_prompt(data_root, task_id, r["x"], shots) for r in rows]
    y_trues = [r["y"] for r in rows]

    for idxs in batched(list(range(len(prompts))), eval_batch_size):
        ps = [prompts[i] for i in idxs]
        ys = [y_trues[i] for i in idxs]
        full_texts = generate_batch(
            model, tok, ps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for full, y_true in zip(full_texts, ys):
            y_pred = parse_output_text(full)
            if strict_vocab_filter:
                y_pred, ok = filter_to_allowed_tokens(y_pred, allowed)
            else:
                toks = split_tokens(y_pred)
                ok = all(t in allowed for t in toks) if toks else True
            parse_ok += int(ok)
            rewards.append(token_edit_reward(y_true, y_pred))
            exact += int(y_pred == y_true)

    n_eval = len(rewards)
    return {
        "n": float(n_eval),
        "mean_reward": float(sum(rewards) / max(n_eval, 1)),
        "exact_rate": float(exact / max(n_eval, 1)),
        "parse_ok_rate": float(parse_ok / max(n_eval, 1)),
    }


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sequential LoRA continual-learning baseline for RuleWorld."
    )

    # I/O
    ap.add_argument("--manifest", type=str, required=True, help="Path to ruleworld_manifest_v1.json")
    ap.add_argument("--data_root", type=str, required=True, help="Root dataset directory (e.g. data/ruleworld_v1)")
    ap.add_argument("--model", type=str, required=True, help="HuggingFace model name or local path")

    # Task stream
    ap.add_argument("--max_tasks", type=int, default=0, help="If >0, only run first K tasks (useful for debugging).")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="eval")
    ap.add_argument("--retention_split", type=str, default="retention")
    ap.add_argument("--do_retention_eval", action="store_true",
                    help="Also evaluate on retention_split after each task.")
    ap.add_argument("--n_eval", type=int, default=50,
                    help="Max eval rows per task after each training task.")
    ap.add_argument("--n_retention", type=int, default=20,
                    help="Max retention rows per task (only used if --do_retention_eval).")
    ap.add_argument("--train_shots", type=int, default=0,
                    help="Number of in-context examples prepended to each training prompt.")
    ap.add_argument("--eval_shots", type=int, default=4,
                    help="Number of in-context examples prepended to each eval prompt.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Decoding temperature during eval (0 = greedy).")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=32)

    # Training budget
    ap.add_argument("--steps_per_task", type=int, default=200,
                    help="Number of gradient-update steps per task.")
    ap.add_argument("--train_batch_size", type=int, default=2,
                    help="Micro-batch size per gradient accumulation step.")
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="Gradient accumulation steps (effective batch = train_batch_size * grad_accum).")
    ap.add_argument("--max_seq_len", type=int, default=512,
                    help="Maximum token length for prompt+completion during training.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0,
                    help="Max gradient norm for clipping (<=0 disables).")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling.")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="",
                    help="Comma-separated list of module names to apply LoRA to. "
                         "If empty, auto-detected from the model architecture.")

    # Eval
    ap.add_argument("--eval_batch_size", type=int, default=2)

    # Output / resume
    ap.add_argument("--run_dir", type=str, default="",
                    help="Explicit output directory. Auto-generated under runs/ if not set.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume training from the latest checkpoint in --run_dir.")

    # Misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--require_cuda", action="store_true",
                    help="Exit with an error if CUDA is unavailable.")
    ap.add_argument("--strict_vocab_filter", action="store_true",
                    help="Filter model outputs to the allowed RuleWorld vocabulary before scoring.")
    args = ap.parse_args()

    set_seed(args.seed)

    # Device check
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. You may have installed CPU-only PyTorch.\n"
            'Verify with: python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"'
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load manifest
    manifest = load_manifest(args.manifest)
    manifest_hash = sha256_file(args.manifest)
    allowed = allowed_token_set(manifest)

    tasks = sorted(manifest["tasks"], key=lambda t: int(t["index"]))
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    task_ids = [t["task_id"] for t in tasks]

    # Output folder
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
        run_id = os.path.basename(run_dir.rstrip("/\\"))
    else:
        stamp = now_stamp()
        run_id = (
            f"lora_seq_{safe_model_name(args.model)}_steps{args.steps_per_task}"
            f"_bs{args.train_batch_size}x{args.grad_accum}_evaln{args.n_eval}_{stamp}"
        )
        run_dir = os.path.join("runs", run_id)
        os.makedirs(run_dir, exist_ok=True)

    # Tokenizer — left-padding is required for correct batched generation
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
    )

    # LoRA setup
    if args.target_modules.strip():
        target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    else:
        target_modules = detect_lora_target_modules(base_model)
        print(f"[INFO] Auto-detected LoRA target modules: {target_modules}", flush=True)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Only optimize trainable (LoRA) parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Mixed-precision (use new-style API; falls back gracefully on older torch)
    use_amp = (device == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        amp_autocast = lambda: torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp)
    except TypeError:
        # torch < 2.x compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        amp_autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16)

    # File paths
    config_path      = os.path.join(run_dir, "config.json")
    state_path       = os.path.join(run_dir, "state.json")
    train_log_path   = os.path.join(run_dir, "train_log.csv")
    eval_matrix_path = os.path.join(run_dir, "eval_matrix.csv")
    aggregates_path  = os.path.join(run_dir, "aggregates.csv")
    summary_path     = os.path.join(run_dir, "summary.json")

    # Write config once
    if not os.path.exists(config_path) or not args.resume:
        cfg = {
            "run_id": run_id,
            "run_dir": run_dir,
            "args": vars(args),
            "manifest_path": args.manifest,
            "manifest_sha256": manifest_hash,
            "git_commit": try_git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "target_modules": target_modules,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    # Resume logic
    start_task_idx = 0
    best_by_task: Dict[str, float] = {}
    if args.resume and os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            start_task_idx = int(st.get("last_completed_task_idx", -1)) + 1
            best_by_task = {k: float(v) for k, v in (st.get("best_by_task") or {}).items()}
            last_task_id = st.get("last_completed_task_id")
            if last_task_id:
                ckpt_dir = os.path.join(run_dir, f"adapter_after_{last_task_id}")
                if os.path.isdir(ckpt_dir):
                    try:
                        model.load_adapter(ckpt_dir, adapter_name="default")
                        model.set_adapter("default")
                        model.enable_adapter_layers()
                        print(f"[INFO] Resumed adapter from {ckpt_dir}", flush=True)
                    except Exception as e:
                        print(f"[WARN] Could not load adapter checkpoint: {e}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not parse resume state, starting from scratch: {e}", flush=True)
            start_task_idx = 0
            best_by_task = {}

    # CSV headers (written once if file doesn't exist)
    def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    ensure_csv_header(train_log_path, [
        "train_task_idx", "train_task_id", "global_step",
        "loss", "lr", "wall_time_sec", "examples_seen",
    ])
    ensure_csv_header(eval_matrix_path, [
        "after_train_task_idx", "after_train_task_id",
        "eval_task_id", "split", "n",
        "mean_reward", "exact_rate", "parse_ok_rate",
        "best_so_far", "forgetting_vs_best",
        "wall_time_sec", "examples_per_sec",
    ])
    ensure_csv_header(aggregates_path, [
        "after_train_task_idx", "after_train_task_id",
        "split", "seen_tasks",
        "avg_mean_reward", "avg_exact_rate", "avg_parse_ok_rate",
        "avg_forgetting", "wall_time_sec",
    ])

    def load_split_rows(task_id: str, split: str, n: int) -> List[Dict[str, Any]]:
        p = os.path.join(args.data_root, task_id, f"{split}.jsonl")
        if not os.path.exists(p):
            return []
        return read_jsonl(p)[:n]

    global_step = 0
    t0_run = time.time()

    for ti in range(start_task_idx, len(task_ids)):
        train_task_id = task_ids[ti]
        print(f"\n=== TRAIN TASK {ti + 1}/{len(task_ids)}: {train_task_id} ===", flush=True)

        train_rows = load_split_rows(train_task_id, args.train_split, n=10_000_000)
        if not train_rows:
            print(f"[WARN] Missing/empty train split for {train_task_id}; skipping training.", flush=True)
        else:
            model.train()
            task_start = time.time()
            examples_seen = 0

            train_prompts = []
            train_completions = []
            for r in train_rows:
                train_prompts.append(build_prompt(args.data_root, train_task_id, r["x"], args.train_shots))
                train_completions.append(r["y"])

            idx = 0
            num = len(train_prompts)
            opt.zero_grad(set_to_none=True)

            for step in range(args.steps_per_task):
                step_loss = 0.0
                step_start = time.time()

                for _ga in range(args.grad_accum):
                    mb_prompts = []
                    mb_completions = []
                    for _ in range(args.train_batch_size):
                        mb_prompts.append(train_prompts[idx])
                        mb_completions.append(train_completions[idx])
                        idx = (idx + 1) % num

                    batch = make_supervised_batch(
                        tok, mb_prompts, mb_completions, max_seq_len=args.max_seq_len
                    )
                    batch = TrainBatch(
                        input_ids=batch.input_ids.to(model.device),
                        attention_mask=batch.attention_mask.to(model.device),
                        labels=batch.labels.to(model.device),
                    )

                    with amp_autocast():
                        out = model(
                            input_ids=batch.input_ids,
                            attention_mask=batch.attention_mask,
                            labels=batch.labels,
                        )
                        loss = out.loss / float(args.grad_accum)

                    scaler.scale(loss).backward()
                    step_loss += float(loss.detach().cpu().item())
                    examples_seen += args.train_batch_size

                if args.clip_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                wall = time.time() - step_start
                with open(train_log_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=[
                        "train_task_idx", "train_task_id", "global_step",
                        "loss", "lr", "wall_time_sec", "examples_seen",
                    ]).writerow({
                        "train_task_idx": ti,
                        "train_task_id": train_task_id,
                        "global_step": global_step,
                        "loss": round(step_loss, 6),
                        "lr": args.lr,
                        "wall_time_sec": round(wall, 4),
                        "examples_seen": examples_seen,
                    })

                if (step + 1) % 25 == 0:
                    print(f"[train {train_task_id}] step {step+1}/{args.steps_per_task} loss={step_loss:.4f}", flush=True)

            train_wall = time.time() - task_start
            print(f"[TRAIN DONE] {train_task_id} wall_time_sec={train_wall:.1f}", flush=True)

        # Save adapter checkpoint after each task
        ckpt_dir = os.path.join(run_dir, f"adapter_after_{train_task_id}")
        try:
            model.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"[WARN] Could not save adapter checkpoint to {ckpt_dir}: {e}", flush=True)

        # Evaluate on all tasks seen so far
        seen_task_ids = task_ids[: ti + 1]

        def run_eval_suite(split_name: str, n_rows: int) -> Dict[str, float]:
            suite_start = time.time()
            per_task_metrics: Dict[str, Dict[str, float]] = {}

            for eval_task_id in seen_task_ids:
                eval_rows = load_split_rows(eval_task_id, split_name, n=n_rows)
                if not eval_rows:
                    continue

                ev_start = time.time()
                m = eval_task_rows(
                    model=model, tok=tok,
                    data_root=args.data_root, task_id=eval_task_id,
                    rows=eval_rows, shots=args.eval_shots,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    allowed=allowed, strict_vocab_filter=args.strict_vocab_filter,
                    eval_batch_size=args.eval_batch_size,
                )
                ev_wall = time.time() - ev_start
                exps = float(m["n"]) / max(ev_wall, 1e-9)

                mean_r = float(m["mean_reward"])
                prev_best = best_by_task.get(eval_task_id, float("-inf"))
                best_now = max(prev_best, mean_r)
                best_by_task[eval_task_id] = best_now
                forgetting = (best_now - mean_r) if math.isfinite(best_now) else float("nan")

                with open(eval_matrix_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=[
                        "after_train_task_idx", "after_train_task_id",
                        "eval_task_id", "split", "n",
                        "mean_reward", "exact_rate", "parse_ok_rate",
                        "best_so_far", "forgetting_vs_best",
                        "wall_time_sec", "examples_per_sec",
                    ]).writerow({
                        "after_train_task_idx": ti,
                        "after_train_task_id": train_task_id,
                        "eval_task_id": eval_task_id,
                        "split": split_name,
                        "n": int(m["n"]),
                        "mean_reward": round(mean_r, 6),
                        "exact_rate": round(float(m["exact_rate"]), 6),
                        "parse_ok_rate": round(float(m["parse_ok_rate"]), 6),
                        "best_so_far": round(best_now, 6) if math.isfinite(best_now) else "nan",
                        "forgetting_vs_best": round(forgetting, 6) if math.isfinite(forgetting) else "nan",
                        "wall_time_sec": round(ev_wall, 4),
                        "examples_per_sec": round(exps, 4),
                    })

                per_task_metrics[eval_task_id] = {
                    "mean_reward": mean_r,
                    "exact_rate": float(m["exact_rate"]),
                    "parse_ok_rate": float(m["parse_ok_rate"]),
                    "forgetting": forgetting,
                }
                print(
                    f"[EVAL {split_name}] after {train_task_id} on {eval_task_id}: "
                    f"mean_reward={mean_r:.4f} forget={forgetting:.4f}", flush=True
                )

            vals = list(per_task_metrics.values())
            if not vals:
                return {
                    "avg_mean_reward": float("nan"),
                    "avg_exact_rate": float("nan"),
                    "avg_parse_ok_rate": float("nan"),
                    "avg_forgetting": float("nan"),
                    "wall_time_sec": time.time() - suite_start,
                }

            finite_forgetting = [v["forgetting"] for v in vals if math.isfinite(v["forgetting"])]
            return {
                "avg_mean_reward": sum(v["mean_reward"] for v in vals) / len(vals),
                "avg_exact_rate": sum(v["exact_rate"] for v in vals) / len(vals),
                "avg_parse_ok_rate": sum(v["parse_ok_rate"] for v in vals) / len(vals),
                "avg_forgetting": sum(finite_forgetting) / len(finite_forgetting) if finite_forgetting else float("nan"),
                "wall_time_sec": time.time() - suite_start,
            }

        def write_agg_row(agg: Dict[str, float], split_name: str) -> None:
            with open(aggregates_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=[
                    "after_train_task_idx", "after_train_task_id",
                    "split", "seen_tasks",
                    "avg_mean_reward", "avg_exact_rate", "avg_parse_ok_rate",
                    "avg_forgetting", "wall_time_sec",
                ]).writerow({
                    "after_train_task_idx": ti,
                    "after_train_task_id": train_task_id,
                    "split": split_name,
                    "seen_tasks": len(seen_task_ids),
                    "avg_mean_reward":    round(float(agg["avg_mean_reward"]),    6) if math.isfinite(agg["avg_mean_reward"])    else "nan",
                    "avg_exact_rate":     round(float(agg["avg_exact_rate"]),     6) if math.isfinite(agg["avg_exact_rate"])     else "nan",
                    "avg_parse_ok_rate":  round(float(agg["avg_parse_ok_rate"]),  6) if math.isfinite(agg["avg_parse_ok_rate"])  else "nan",
                    "avg_forgetting":     round(float(agg["avg_forgetting"]),     6) if math.isfinite(agg["avg_forgetting"])     else "nan",
                    "wall_time_sec":      round(float(agg["wall_time_sec"]),      4),
                })

        eval_agg = run_eval_suite(args.eval_split, args.n_eval)
        write_agg_row(eval_agg, args.eval_split)

        if args.do_retention_eval:
            ret_agg = run_eval_suite(args.retention_split, args.n_retention)
            write_agg_row(ret_agg, args.retention_split)

        # Persist resume state
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({
                "last_completed_task_idx": ti,
                "last_completed_task_id": train_task_id,
                "best_by_task": best_by_task,
            }, f, indent=2)

    # Final summary
    total_wall = time.time() - t0_run
    summary = {
        "run_id": run_id,
        "run_dir": run_dir,
        "manifest_sha256": manifest_hash,
        "model": args.model,
        "tasks": task_ids,
        "total_wall_time_sec": total_wall,
        "best_by_task": best_by_task,
        "train_log_csv": train_log_path,
        "eval_matrix_csv": eval_matrix_path,
        "aggregates_csv": aggregates_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SEQUENTIAL LORA BASELINE COMPLETE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
