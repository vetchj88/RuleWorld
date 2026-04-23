# ruleworld/dataset_builder.py
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional

from ruleworld.task_loader import TaskRegistry, load_registry, detokenize


# ----------------------------
# Deterministic seeding (matches your manifest meta string)
# seed32(s)=uint32_from_sha256(s) using first 4 bytes big-endian
# ----------------------------

def seed32(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="big", signed=False)

def task_seed(meta: Dict[str, Any], index: int) -> int:
    version = meta["ruleworld_version"]
    gseed = int(meta["global_seed"])
    return seed32(f"{version}|GLOBAL|{gseed}|TASK|{index:02d}")

def split_seed(meta: Dict[str, Any], index: int, split: str) -> int:
    ts = task_seed(meta, index)
    return seed32(f"{ts}|SPLIT|{split.upper()}")


# ----------------------------
# Generation config (world-level constants)
# These do NOT re-encode task parameters; those come from the manifest.
# ----------------------------

@dataclass(frozen=True)
class GenConfig:
    # default lengths
    sym_Lmin: int = 6
    sym_Lmax: int = 18
    dig_Lmin: int = 8
    dig_Lmax: int = 20

    # special bounding for MIRROR-heavy tasks (keeps outputs sane)
    mirror_Lmin: int = 4
    mirror_Lmax: int = 10

    # constraints
    min_hits_in_A: int = 2
    min_output_len: int = 2
    max_output_len: int = 30


# ----------------------------
# Input generators
# ----------------------------

def gen_symbol_seq(rng: random.Random, vocab_S: Sequence[str], Lmin: int, Lmax: int) -> List[str]:
    L = rng.randint(Lmin, Lmax)
    return [rng.choice(vocab_S) for _ in range(L)]

def gen_digit_seq(rng: random.Random, vocab_D: Sequence[str], Lmin: int, Lmax: int) -> List[str]:
    L = rng.randint(Lmin, Lmax)
    return [rng.choice(vocab_D) for _ in range(L)]

def gen_runlen_symbol_seq(rng: random.Random, vocab_S: Sequence[str], Lmin: int, Lmax: int) -> List[str]:
    """
    Generate as concatenated runs, each run length 1..9, total length in [Lmin,Lmax].
    """
    target_L = rng.randint(Lmin, Lmax)
    out: List[str] = []
    while len(out) < target_L:
        t = rng.choice(vocab_S)
        remaining = target_L - len(out)
        r = rng.randint(1, min(9, remaining))
        out.extend([t] * r)
    return out


# ----------------------------
# Task-aware rejection sampling
# (uses manifest ops + params, but does not invent params)
# ----------------------------

def _count_hits(x: Sequence[str], A: Sequence[str]) -> int:
    Aset = set(A)
    return sum(1 for t in x if t in Aset)

def _preferred_symbol_length_bounds(task_op: str, pipeline_ops: Sequence[str], cfg: GenConfig) -> Tuple[int, int]:
    # If MIRROR appears anywhere, bounding input helps keep outputs < max_output_len.
    if task_op == "MIRROR" or "MIRROR" in pipeline_ops:
        return cfg.mirror_Lmin, cfg.mirror_Lmax
    return cfg.sym_Lmin, cfg.sym_Lmax

def sample_valid_example(
    rng: random.Random,
    reg: TaskRegistry,
    task_id: str,
    cfg: GenConfig,
    max_tries: int = 10_000,
) -> Tuple[List[str], List[str]]:
    """
    Rejection-sample an (x_tokens, y_tokens) pair that satisfies:
    - task-specific non-triviality (min hits, min output len)
    - global bounds (max output len)
    - domain-appropriate inputs (no marker by default)
    """
    task = reg.get(task_id)
    vocab_S = reg.vocab.S
    vocab_D = reg.vocab.D

    # Determine op/pipeline for heuristics
    if hasattr(task, "op"):
        task_op = getattr(task, "op")
        pipeline_ops: List[str] = []
        params_for_first_op = getattr(task, "params", {})
    else:
        task_op = "COMPOSITE"
        pipeline = getattr(task, "pipeline", [])
        pipeline_ops = [op for op, _ in pipeline]
        params_for_first_op = pipeline[0][1] if pipeline else {}

    for _ in range(max_tries):
        # 1) Generate candidate input x
        if task.domain == "S":
            Lmin, Lmax = _preferred_symbol_length_bounds(task_op, pipeline_ops, cfg)

            # Special handling: RUNLEN wants run-structured input
            first_op = task_op if task_op != "COMPOSITE" else (pipeline_ops[0] if pipeline_ops else "")
            if first_op == "RUNLEN":
                x = gen_runlen_symbol_seq(rng, vocab_S, Lmin, Lmax)
            else:
                # SLICE needs L >= p+w if SLICE is first op
                if first_op == "SLICE" and "p" in params_for_first_op and "w" in params_for_first_op:
                    p = int(params_for_first_op["p"])
                    w = int(params_for_first_op["w"])
                    Lmin2 = max(Lmin, p + w)
                    Lmax2 = max(Lmax, p + w)
                    x = gen_symbol_seq(rng, vocab_S, Lmin2, Lmax2)
                else:
                    x = gen_symbol_seq(rng, vocab_S, Lmin, Lmax)

            # Enforce "min hits" for ops depending on A
            # (FILTERIN, DUPLICATE, INSERTZZ)
            needs_A_hits = False
            A: Optional[Sequence[str]] = None

            def scan_for_A(op: str, params: Dict[str, Any]) -> Optional[Sequence[str]]:
                if op in ("FILTERIN", "DUPLICATE", "INSERTZZ", "PARTITION") and "A" in params:
                    return params["A"]
                return None

            if task_op != "COMPOSITE":
                A = scan_for_A(task_op, task.params)
            else:
                # look at each step in pipeline for A-bearing ops
                for op, p in getattr(task, "pipeline", []):
                    maybeA = scan_for_A(op, p)
                    if maybeA is not None:
                        A = maybeA
                        break

            if A is not None:
                needs_A_hits = True

            if needs_A_hits and _count_hits(x, A) < cfg.min_hits_in_A:
                continue

        elif task.domain == "D":
            x = gen_digit_seq(rng, vocab_D, cfg.dig_Lmin, cfg.dig_Lmax)
        else:
            # Unknown domain string: fall back based on token type expectation
            # (You can tighten this later)
            x = gen_symbol_seq(rng, vocab_S, cfg.sym_Lmin, cfg.sym_Lmax)

        # 2) Apply task transform
        y = task.apply(x)

        # 3) Global constraints
        if len(y) < cfg.min_output_len:
            continue
        if len(y) > cfg.max_output_len:
            continue

        # 4) Task-specific constraints: STRIDEKEEP and FILTERIN can produce tiny outputs
        # (already covered by min_output_len) but you can tighten here if desired.

        return x, y

    raise RuntimeError(f"Failed to sample valid example for {task_id} after {max_tries} tries.")


# ----------------------------
# Split builders
# ----------------------------

def build_split(
    meta: Dict[str, Any],
    reg: TaskRegistry,
    task_index: int,
    task_id: str,
    split: str,
    n: int,
    cfg: GenConfig,
    avoid_inputs: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(split_seed(meta, task_index, split))
    out: List[Dict[str, Any]] = []
    seen = set() if avoid_inputs is None else avoid_inputs

    while len(out) < n:
        x_toks, y_toks = sample_valid_example(rng, reg, task_id, cfg)
        x_str = detokenize(x_toks)
        if x_str in seen:
            continue
        seen.add(x_str)

        out.append({
            "task_id": task_id,
            "index": task_index,
            "split": split.lower(),
            "x": x_str,
            "y": detokenize(y_toks),
            "x_tokens": x_toks,
            "y_tokens": y_toks,
        })
    return out

def build_all_task_splits(
    manifest: Dict[str, Any],
    reg: TaskRegistry,
    cfg: GenConfig,
    exemplar_n: int = 64,
    eval_n: int = 1000,
    retention_n: int = 200,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    meta = manifest["meta"]
    tasks = sorted(manifest["tasks"], key=lambda t: int(t["index"]))

    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for t in tasks:
        idx = int(t["index"])
        tid = t["task_id"]

        # ensure no input overlap across splits for this task
        seen_inputs: set[str] = set()

        exemplar = build_split(meta, reg, idx, tid, "EXEMPLAR", exemplar_n, cfg, avoid_inputs=seen_inputs)
        evalset  = build_split(meta, reg, idx, tid, "EVAL", eval_n, cfg, avoid_inputs=seen_inputs)
        retset   = build_split(meta, reg, idx, tid, "RETENTION", retention_n, cfg, avoid_inputs=seen_inputs)

        result[tid] = {"exemplar": exemplar, "eval": evalset, "retention": retset}

    return result


# ----------------------------
# Writers
# ----------------------------

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def materialize_dataset_tree(out_root: str, all_splits: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
    for task_id, splits in all_splits.items():
        task_dir = os.path.join(out_root, task_id)
        write_jsonl(os.path.join(task_dir, "exemplar.jsonl"), splits["exemplar"])
        write_jsonl(os.path.join(task_dir, "eval.jsonl"), splits["eval"])
        write_jsonl(os.path.join(task_dir, "retention.jsonl"), splits["retention"])


# ----------------------------
# Convenience: load manifest + registry and build
# ----------------------------

def build_and_write(manifest_path: str, out_root: str) -> None:
    # Load registry using your loader (single source of truth for task params)
    reg = load_registry(manifest_path)

    # Load manifest dict (we need meta + tasks list)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cfg = GenConfig()
    all_splits = build_all_task_splits(manifest, reg, cfg)
    materialize_dataset_tree(out_root, all_splits)
