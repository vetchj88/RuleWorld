"""
ruleworld/task_loader.py

Loads the RuleWorld manifest you uploaded (meta + tasks with primitive/composite).
Manifest fields used:
- manifest["meta"]["vocab_S"], ["vocab_D"], ["marker_token"]
- manifest["tasks"][i]["task_id"], ["type"], ["domain"]
- primitive: ["op"], ["params"]
- composite: ["composition"]["f"], ["f_params"], ["g"], ["g_params"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Callable, Tuple, Optional
import json
import os


# ----------------------------
# IO
# ----------------------------

def _maybe_load_yaml(path: str) -> Optional[Dict[str, Any]]:
    if not path.lower().endswith((".yaml", ".yml")):
        return None
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "YAML manifest provided but PyYAML is not installed. "
            "Install with: pip install pyyaml"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_manifest(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    y = _maybe_load_yaml(path)
    if y is not None:
        return y

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Token helpers
# ----------------------------

def tokenize(seq: str) -> List[str]:
    seq = seq.strip()
    return [t for t in seq.split(" ") if t] if seq else []

def detokenize(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


# ----------------------------
# Vocab / Tasks
# ----------------------------

@dataclass(frozen=True)
class Vocab:
    S: List[str]
    D: List[str]
    marker: str  # "zz"

    @property
    def allowed_tokens(self) -> set[str]:
        # union of all known tokens; RUNLEN outputs a mix of S + D
        return set(self.S) | set(self.D) | {self.marker}

    def is_symbol(self, t: str) -> bool:
        return t in set(self.S) or t == self.marker

    def is_digit(self, t: str) -> bool:
        return t in set(self.D)

@dataclass
class Task:
    task_id: str
    kind: str      # "primitive" or "composite"
    domain: str    # "S", "D", "S→(S,D)", etc. (kept verbatim from manifest)
    vocab: Vocab

    def apply(self, x: Sequence[str]) -> List[str]:
        raise NotImplementedError

    def __call__(self, x: Sequence[str]) -> List[str]:
        return self.apply(x)

@dataclass
class PrimitiveTask(Task):
    op: str
    params: Dict[str, Any]

    def apply(self, x: Sequence[str]) -> List[str]:
        fn = PRIMITIVE_OPS.get(self.op)
        if fn is None:
            raise KeyError(f"Unknown primitive op '{self.op}' for task {self.task_id}")
        y = fn(list(x), self.params, self.vocab)
        return y

@dataclass
class CompositeTask(Task):
    # pipeline: [("OPNAME", params), ...] applied in order
    pipeline: List[Tuple[str, Dict[str, Any]]]

    def apply(self, x: Sequence[str]) -> List[str]:
        y = list(x)
        for op, params in self.pipeline:
            fn = PRIMITIVE_OPS.get(op)
            if fn is None:
                raise KeyError(f"Unknown op '{op}' in composite task {self.task_id}")
            y = fn(y, params, self.vocab)
        return y


# ----------------------------
# Primitive ops (match manifest op strings)
# ----------------------------

def _rev(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    return list(reversed(x))

def _rotl(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    k = int(params["k"])
    if not x:
        return []
    k = k % len(x)
    return x[k:] + x[:k]

def _chunkrev(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    g = int(params["g"])
    if g <= 0:
        raise ValueError("CHUNKREV requires g >= 1")
    out: List[str] = []
    for i in range(0, len(x), g):
        out.extend(reversed(x[i:i+g]))
    return out

def _stridekeep(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    s = int(params["s"])
    o = int(params["o"])
    if s <= 0:
        raise ValueError("STRIDEKEEP requires s >= 1")
    return [t for j, t in enumerate(x) if ((j - o) % s) == 0]

def _filterin(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    A = set(params["A"])
    return [t for t in x if t in A]

def _subst(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    """
    Permutation convention (from manifest meta): base_order[i] -> pi_list[i].
    Here base_order is vocab.S (fixed order).
    """
    pi_list = params["pi_list"]
    if len(pi_list) != len(vocab.S):
        raise ValueError("SUBST requires pi_list length == len(vocab.S)")
    mapping = {vocab.S[i]: pi_list[i] for i in range(len(vocab.S))}
    mapping[vocab.marker] = vocab.marker  # identity for marker
    return [mapping.get(t, t) for t in x]

def _duplicate(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    A = set(params["A"])
    out: List[str] = []
    for t in x:
        out.append(t)
        if t in A:
            out.append(t)
    return out

def _insertzz(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    A = set(params["A"])
    out: List[str] = []
    for t in x:
        out.append(t)
        if t in A:
            out.append(vocab.marker)
    return out

def _partition(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    A = set(params["A"])
    front = [t for t in x if t in A]
    back = [t for t in x if t not in A]
    return front + back

def _sort(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    order = {tok: i for i, tok in enumerate(vocab.S)}
    return sorted(x, key=lambda t: order.get(t, 10**9))

def _unique(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in x:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _mirror(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    return x + list(reversed(x))

def _slice(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    p = int(params["p"])
    w = int(params["w"])
    if p < 0 or w < 0:
        raise ValueError("SLICE requires p,w >= 0")
    return x[p:p+w]

def _runlen(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    """
    RUNLEN: for each run token t repeated r times (1..9), output t then d{r}.
    Uses digit tokens "d1".."d9".
    """
    if not x:
        return []
    out: List[str] = []
    i = 0
    while i < len(x):
        t = x[i]
        j = i + 1
        while j < len(x) and x[j] == t:
            j += 1
        r = j - i
        if r < 1 or r > 9:
            raise ValueError(f"RUNLEN run length {r} out of supported range 1..9")
        out.append(t)
        out.append(f"d{r}")
        i = j
    return out

# Digit ops
def _dadd(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    c = int(params["c"]) % 10
    out: List[str] = []
    for t in x:
        n = int(t[1:])  # "dN"
        out.append(f"d{(n + c) % 10}")
    return out

def _dmul(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    m = int(params["m"]) % 10
    out: List[str] = []
    for t in x:
        n = int(t[1:])
        out.append(f"d{(n * m) % 10}")
    return out

def _dprefixsum(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    out: List[str] = []
    acc = 0
    for i, t in enumerate(x):
        n = int(t[1:])
        if i == 0:
            acc = n % 10
        else:
            acc = (acc + n) % 10
        out.append(f"d{acc}")
    return out

def _dperm(x: List[str], params: Dict[str, Any], vocab: Vocab) -> List[str]:
    """
    sigma_list length 10 maps digit value i -> sigma_list[i]
    """
    sigma_list = params["sigma_list"]
    if len(sigma_list) != 10:
        raise ValueError("DPERM requires sigma_list length == 10")
    out: List[str] = []
    for t in x:
        n = int(t[1:])
        out.append(f"d{int(sigma_list[n])}")
    return out


PRIMITIVE_OPS: Dict[str, Callable[[List[str], Dict[str, Any], Vocab], List[str]]] = {
    # Symbol primitives
    "REV": _rev,
    "ROTL": _rotl,
    "CHUNKREV": _chunkrev,
    "STRIDEKEEP": _stridekeep,
    "FILTERIN": _filterin,
    "SUBST": _subst,
    "DUPLICATE": _duplicate,
    "INSERTZZ": _insertzz,
    "PARTITION": _partition,
    "SORT": _sort,
    "UNIQUE": _unique,
    "MIRROR": _mirror,
    "SLICE": _slice,
    "RUNLEN": _runlen,

    # Digit primitives (some reuse the same underlying functions)
    "DREV": _rev,
    "DROT": _rotl,
    "DADD": _dadd,
    "DMUL": _dmul,
    "DPREFIXSUM": _dprefixsum,
    "DPERM": _dperm,
}


# ----------------------------
# Registry / builder
# ----------------------------

@dataclass
class TaskRegistry:
    vocab: Vocab
    tasks_by_id: Dict[str, Task]
    tasks_in_order: List[Task]

    def get(self, task_id: str) -> Task:
        return self.tasks_by_id[task_id]


def build_registry_from_manifest(manifest: Dict[str, Any]) -> TaskRegistry:
    meta = manifest.get("meta", {})
    vocab = Vocab(
        S=list(meta.get("vocab_S", [])),
        D=list(meta.get("vocab_D", [])),
        marker=str(meta.get("marker_token", "zz")),
    )

    tasks_raw = manifest.get("tasks", [])
    tasks_by_id: Dict[str, Task] = {}
    tasks_in_order: List[Task] = []

    # Sort by index to enforce curriculum order
    tasks_raw_sorted = sorted(tasks_raw, key=lambda td: int(td["index"]))

    for td in tasks_raw_sorted:
        tid = td["task_id"]
        ttype = td["type"]  # "primitive" | "composite"
        domain = td.get("domain", "")

        if ttype == "primitive":
            op = td["op"]
            params = td.get("params", {}) or {}
            if op not in PRIMITIVE_OPS:
                raise KeyError(
                    f"Unknown primitive op '{op}' for {tid}. "
                    f"Known ops: {sorted(PRIMITIVE_OPS.keys())}"
                )
            task = PrimitiveTask(
                task_id=tid,
                kind="primitive",
                domain=domain,
                vocab=vocab,
                op=op,
                params=params,
            )

        elif ttype == "composite":
            comp = td.get("composition")
            if not isinstance(comp, dict):
                raise ValueError(f"Composite task {tid} missing 'composition' dict.")
            f_op = comp["f"]
            g_op = comp["g"]
            f_params = comp.get("f_params", {}) or {}
            g_params = comp.get("g_params", {}) or {}

            # Function composition uses f(g(x)) => apply g first, then f
            pipeline = [(g_op, g_params), (f_op, f_params)]

            for op, _ in pipeline:
                if op not in PRIMITIVE_OPS:
                    raise KeyError(
                        f"Unknown op '{op}' in composite task {tid}. "
                        f"Known ops: {sorted(PRIMITIVE_OPS.keys())}"
                    )

            task = CompositeTask(
                task_id=tid,
                kind="composite",
                domain=domain,
                vocab=vocab,
                pipeline=pipeline,
            )

        else:
            raise ValueError(f"Unknown task type '{ttype}' for {tid}")

        tasks_by_id[tid] = task
        tasks_in_order.append(task)

    return TaskRegistry(vocab=vocab, tasks_by_id=tasks_by_id, tasks_in_order=tasks_in_order)


def load_registry(manifest_path: str) -> TaskRegistry:
    manifest = load_manifest(manifest_path)
    return build_registry_from_manifest(manifest)


# ----------------------------
# Optional: simple validation
# ----------------------------

def validate_task_output(task: Task, x_tokens: Sequence[str], y_tokens: Sequence[str]) -> None:
    """
    Minimal token validity checks:
    - All outputs must be from vocab S/D or marker
    - If domain == "S", output must be subset of (S ∪ {zz})
    - If domain == "D", output must be subset of D
    - If domain includes "S→(S,D)" we allow S∪D (for RUNLEN)
    """
    vocab = task.vocab
    allowed = vocab.allowed_tokens
    bad = [t for t in y_tokens if t not in allowed]
    if bad:
        raise ValueError(f"{task.task_id} produced invalid tokens: {bad}")

    dom = task.domain
    if dom == "S":
        if any((t not in set(vocab.S) and t != vocab.marker) for t in y_tokens):
            raise ValueError(f"{task.task_id} domain S produced non-S tokens: {y_tokens}")
    elif dom == "D":
        if any(t not in set(vocab.D) for t in y_tokens):
            raise ValueError(f"{task.task_id} domain D produced non-D tokens: {y_tokens}")
    # else: allow mixed or custom domains without strict filtering


# ----------------------------
# CLI sanity check
# ----------------------------

def _demo_apply(reg: TaskRegistry, task_id: str, x_str: str) -> None:
    task = reg.get(task_id)
    x = tokenize(x_str)
    y = task.apply(x)
    validate_task_output(task, x, y)
    print(f"Task: {task.task_id} [{task.kind}] domain={task.domain}")
    print(f"IN : {detokenize(x)}")
    print(f"OUT: {detokenize(y)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--task", type=str, default="P01")
    ap.add_argument("--x", type=str, default="ba di ku lo ga be")
    args = ap.parse_args()

    reg = load_registry(args.manifest)
    _demo_apply(reg, args.task, args.x)
