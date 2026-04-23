import json
import os
from typing import List, Dict, Any


PROMPT_HEADER = "You will transform an INPUT sequence into an OUTPUT sequence.\n\n"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_fewshot_prompt(
    data_root: str,
    task_id: str,
    x_query: str,
    n_shots: int = 4,
) -> str:
    """
    Fixed shots: take first n_shots rows from exemplar.jsonl for this task.
    """
    exemplar_path = os.path.join(data_root, task_id, "exemplar.jsonl")
    ex = read_jsonl(exemplar_path)
    if len(ex) < n_shots:
        raise ValueError(f"Not enough exemplars for {task_id}: have {len(ex)}, need {n_shots}")

    parts = [PROMPT_HEADER]
    for i in range(n_shots):
        parts.append(f"INPUT: {ex[i]['x']}\nOUTPUT: {ex[i]['y']}\n\n")

    parts.append(f"INPUT: {x_query}\nOUTPUT:")
    return "".join(parts)
