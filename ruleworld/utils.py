import hashlib
from typing import Sequence, List

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Token-level Levenshtein edit distance between two token lists."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            current_row.append(min(
                previous_row[j + 1] + 1,   # deletion
                current_row[j]      + 1,   # insertion
                previous_row[j]     + (c1 != c2),  # substitution
            ))
        previous_row = current_row

    return previous_row[-1]
