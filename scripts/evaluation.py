"""
RuleWorld Standalone Evaluation Harness

Scores model predictions against RuleWorld ground truth using:
  - Token Edit Reward: normalized edit distance (0.0 = no overlap, 1.0 = exact)
  - Exact Match (EM): fraction of predictions that are token-for-token identical
                      to ground truth after whitespace normalization

Requirements:
    Python 3.8+  (no external dependencies)

Usage:
    python evaluation.py \\
      --predictions path/to/predictions.jsonl \\
      --ground_truth path/to/eval.jsonl

    # Optionally write results to a JSON file:
    python evaluation.py \\
      --predictions path/to/predictions.jsonl \\
      --ground_truth path/to/eval.jsonl \\
      --output_json path/to/results.json

Input formats (JSONL, one record per line):
    ground_truth : {"y": "<target string>", ...}
    predictions  : {"prediction": "<predicted string>", ...}

    If both files contain an "id" or "x" field, alignment is verified
    automatically and an error is raised on mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure the root directory is in sys.path so ruleworld package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ruleworld.utils import levenshtein


# -----------------------
# Scoring primitives
# -----------------------

def split_tokens(text: str) -> List[str]:
    """Split a string into whitespace-delimited tokens, collapsing extra spaces."""
    return [t for t in (text or "").strip().split() if t]


def token_edit_reward(y_true: str, y_pred: str) -> float:
    """
    Normalized token edit reward in [0.0, 1.0].

    reward = 1 - (edit_distance / max(len(y_true_tokens), len(y_pred_tokens)))

    Identical sequences → 1.0. Completely disjoint sequences → 0.0.
    Both empty → 1.0.
    """
    yt = split_tokens(y_true)
    yp = split_tokens(y_pred)
    if not yt and not yp:
        return 1.0
    denom = max(len(yt), len(yp), 1)
    return max(0.0, min(1.0, 1.0 - levenshtein(yt, yp) / denom))


def is_exact_match(y_true: str, y_pred: str) -> bool:
    """
    Exact match after whitespace normalization.

    Uses the same token-splitting as token_edit_reward to ensure
    consistent scoring: e.g. "a  b" and "a b" are considered identical.
    """
    return split_tokens(y_true) == split_tokens(y_pred)


# -----------------------
# Alignment check
# -----------------------

def _alignment_key(record: Dict[str, Any]) -> Optional[str]:
    """Return a stable identity key from a record, if one is available."""
    for field in ("id", "x"):
        if field in record:
            return str(record[field])
    return None


def _check_alignment(gt: Dict[str, Any], pred: Dict[str, Any], idx: int) -> None:
    """Raise ValueError if ground-truth and prediction records don't correspond."""
    gt_key   = _alignment_key(gt)
    pred_key = _alignment_key(pred)
    if gt_key is not None and pred_key is not None and gt_key != pred_key:
        raise ValueError(
            f"Alignment mismatch at line {idx}: "
            f"ground truth key={gt_key!r}, prediction key={pred_key!r}. "
            "Ensure predictions are in the same order as ground truth."
        )


# -----------------------
# Core evaluation
# -----------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{lineno} — {e}") from e
    return records


def evaluate_file(
    predictions_path: str,
    ground_truth_path: str,
    output_json: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Evaluate a predictions file against its ground truth file.

    Returns:
        (mean_reward, exact_match_rate) — both in [0.0, 1.0].

    Raises:
        ValueError  if the files have different lengths or misaligned IDs.
        FileNotFoundError if either path does not exist.
    """
    for path in (ground_truth_path, predictions_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    ground_truths = load_jsonl(ground_truth_path)
    predictions   = load_jsonl(predictions_path)

    if len(ground_truths) != len(predictions):
        raise ValueError(
            f"Line count mismatch — ground truth has {len(ground_truths)} records "
            f"but predictions has {len(predictions)}. "
            "Ensure every input example has exactly one prediction."
        )

    count = len(ground_truths)
    total_reward  = 0.0
    exact_matches = 0

    for i in range(count):
        gt_record   = ground_truths[i]
        pred_record = predictions[i]

        _check_alignment(gt_record, pred_record, i)

        y_true = str(gt_record.get("y", ""))
        y_pred = str(pred_record.get("prediction", ""))

        total_reward  += token_edit_reward(y_true, y_pred)
        exact_matches += int(is_exact_match(y_true, y_pred))

    mean_reward = total_reward  / count if count > 0 else 0.0
    em_rate     = exact_matches / count if count > 0 else 0.0

    # Console output
    label = os.path.basename(predictions_path)
    print(f"Results for {label}:")
    print(f"  Total Samples : {count}")
    print(f"  Mean Reward   : {mean_reward:.4f}")
    print(f"  Exact Match   : {em_rate:.4f}")
    print("-" * 40)

    # Optional JSON output
    if output_json:
        results = {
            "predictions_file":  predictions_path,
            "ground_truth_file": ground_truth_path,
            "n":                 count,
            "mean_reward":       round(mean_reward, 6),
            "exact_match_rate":  round(em_rate,     6),
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_json}")

    return mean_reward, em_rate


# -----------------------
# CLI
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions on RuleWorld datasets."
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to the JSONL file containing model predictions.",
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True,
        help="Path to the JSONL ground truth file.",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Optional path to write results as a JSON file.",
    )
    args = parser.parse_args()

    try:
        mean_reward, em_rate = evaluate_file(
            predictions_path=args.predictions,
            ground_truth_path=args.ground_truth,
            output_json=args.output_json,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)