"""
RuleWorld Dataset Generator
Generates training, evaluation, and retention sets based on the RuleWorld manifest.
Ensures zero data contamination (train sets do not overlap with eval/retention sets).
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, List, Set

# Ensure the root directory is in sys.path so ruleworld package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Imports from the local ruleworld module
from ruleworld.dataset_builder import build_and_write
from ruleworld.prompting import read_jsonl
from ruleworld.task_loader import load_registry
from ruleworld.utils import sha256_file

def _split_tokens(s: str) -> List[str]:
    return [t for t in (s or "").strip().split(" ") if t]

def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _read_x_set(path: str) -> Set[str]:
    """Reads a jsonl file and returns a set of input strings for overlap avoidance."""
    xs: Set[str] = set()
    if not path or not os.path.exists(path):
        return xs
    for r in read_jsonl(path):
        x = r.get("x")
        if isinstance(x, str) and x.strip():
            xs.add(" ".join(_split_tokens(x)))
        elif isinstance(r.get("x_tokens"), list):
            xs.add(" ".join([str(t) for t in r["x_tokens"] if str(t).strip()]))
    return xs

def _eval_len_pool(task_dir: str, split: str = "eval") -> List[int]:
    """Builds an empirical length pool from existing eval/retention splits."""
    path = os.path.join(task_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        return []
    lens: List[int] = []
    for r in read_jsonl(path):
        if isinstance(r.get("x_tokens"), list) and r["x_tokens"]:
            lens.append(len(r["x_tokens"]))
        else:
            lens.append(len(_split_tokens(r.get("x", ""))))
    return [L for L in lens if L > 0]

def _sample_x_tokens(rng: random.Random, vocab: List[str], length: int) -> List[str]:
    return [rng.choice(vocab) for _ in range(length)]

def generate_train_splits(args, reg):
    """Generates the training splits ensuring no overlap with eval/retention."""
    vocab_S: List[str] = reg.vocab.S
    vocab_D: List[str] = reg.vocab.D

    only_tasks = [t.strip() for t in args.only_tasks.split(",")] if args.only_tasks else None

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta = {
        "generated_at": stamp,
        "manifest_path": args.manifest,
        "manifest_sha256": sha256_file(args.manifest),
        "train_size": args.train_size,
        "seed": args.seed
    }

    ok_count = 0
    
    for task_id, task in reg.tasks_by_id.items():
        if only_tasks and task_id not in only_tasks:
            continue

        task_dir = os.path.join(args.out, task_id)
        _mkdir(task_dir)

        out_path = os.path.join(task_dir, "train.jsonl")
        meta_path = os.path.join(task_dir, "train_meta.json")

        if os.path.exists(out_path) and not args.overwrite:
            print(f"[SKIP] {task_id}: train.jsonl exists (use --overwrite to regenerate)")
            continue

        # Get length distribution from eval set
        len_pool = _eval_len_pool(task_dir, split="eval") or _eval_len_pool(task_dir, split="retention")

        # Avoid overlap with eval/retention sets
        used_x: Set[str] = set()
        used_x |= _read_x_set(os.path.join(task_dir, "eval.jsonl"))
        used_x |= _read_x_set(os.path.join(task_dir, "retention.jsonl"))

        domain = str(getattr(task, "domain", "") or "")
        vocab = vocab_S if "S" in domain else vocab_D

        # Deterministic RNG per task
        h = int(hashlib.sha256(f"{args.seed}:{task_id}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(h)

        rows: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = args.train_size * 50

        while len(rows) < args.train_size:
            attempts += 1
            if attempts > max_attempts:
                print(f"[ERROR] Failed to generate enough unique rows for {task_id}.")
                break

            L = rng.choice(len_pool) if len_pool else rng.randint(3, 12)
            x_tokens = _sample_x_tokens(rng, vocab, L)
            x_str = " ".join(x_tokens)

            # Deduplication
            if x_str in used_x:
                continue
            used_x.add(x_str)

            try:
                y_tokens = task.apply(x_tokens)
                y_str = " ".join([str(t) for t in y_tokens if str(t).strip()])
                rows.append({"x": x_str, "y": y_str})
            except Exception:
                continue

        # Write data
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Write metadata
        task_meta = dict(run_meta)
        task_meta.update({"task_id": task_id, "domain": domain, "rows": len(rows)})
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(task_meta, f, indent=2)

        ok_count += 1
        print(f"[OK] {task_id}: wrote {len(rows)} training rows.")

    if ok_count == 0:
        print("[WARN] No training splits generated.")

def main():
    ap = argparse.ArgumentParser(description="Generate RuleWorld benchmark datasets.")
    ap.add_argument("--manifest", type=str, required=True, help="Path to ruleworld_manifest_v1.json")
    ap.add_argument("--out", type=str, required=True, help="Output root directory, e.g. data/ruleworld_v1")
    ap.add_argument("--train_size", type=int, default=2000, help="Number of training rows to generate per task")
    ap.add_argument("--seed", type=int, default=1337, help="Global RNG seed for reproducibility")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing datasets")
    ap.add_argument("--only_tasks", type=str, default="", help="Limit to specific tasks (e.g. P01,C01)")
    args = ap.parse_args()

    print("Step 1: Building Base Evaluation & Retention Sets...")
    build_and_write(args.manifest, args.out)

    print("\nStep 2: Generating De-duplicated Training Splits...")
    registry = load_registry(args.manifest)
    generate_train_splits(args, registry)

    print(f"\nRuleWorld Generation Complete! Data saved to: {args.out}")

if __name__ == "__main__":
    main()