import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from rag.embeddings import Embedder
from rag.models import Chunk, Hit, RagError
from rag.search import MODES, Mode, search
from rag.store import Store

SYNTHETIC_NOTE = (
    "note: synthetic queries copy a heading or line from the target chunk, so they are easier "
    "than real questions and favour keyword search. Do not read them as a quality estimate; "
    "use them to catch regressions between runs."
)


@dataclass(frozen=True)
class Case:
    query: str
    expected: tuple[str, ...]
    synthetic: bool = False


@dataclass(frozen=True)
class Metrics:
    hit_at_1: float
    hit_at_k: float
    mrr: float


def load_cases(path: Path) -> list[Case]:
    if not path.is_file():
        raise RagError(f"eval file not found: {path}")
    cases = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            query, expected = row["query"], row["expected"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RagError(f"{path}:{n}: expected {{\"query\": str, \"expected\": [str, ...]}} ({exc})") from exc
        if not isinstance(query, str) or not isinstance(expected, list) or not expected:
            raise RagError(f"{path}:{n}: query must be a string and expected a non-empty list")
        cases.append(Case(query, tuple(expected), bool(row.get("synthetic", False))))
    if not cases:
        raise RagError(f"{path} has no cases")
    return cases


def is_relevant(chunk: Chunk, expected: tuple[str, ...]) -> bool:
    return any(chunk.path == e or chunk.path.startswith(e) for e in expected)


def first_relevant_rank(hits: list[Hit], expected: tuple[str, ...]) -> int | None:
    return next((rank for rank, hit in enumerate(hits, 1) if is_relevant(hit.chunk, expected)), None)


def metrics(ranks: list[int | None], k: int) -> Metrics:
    """Aggregate first-relevant ranks (None = no relevant hit in the top k)."""
    n = len(ranks)
    within = [r for r in ranks if r is not None and r <= k]
    return Metrics(
        hit_at_1=sum(1 for r in within if r == 1) / n,
        hit_at_k=len(within) / n,
        mrr=sum(1 / r for r in within) / n,
    )


def evaluate(
    cases: list[Case], store: Store, embedder: Embedder, k: int, modes: tuple[Mode, ...] = MODES
) -> dict[Mode, list[int | None]]:
    return {
        mode: [first_relevant_rank(search(c.query, store, embedder, k=k, mode=mode), c.expected) for c in cases]
        for mode in modes
    }


def report(cases: list[Case], ranks: dict[Mode, list[int | None]], k: int) -> str:
    modes = list(ranks)
    out = [f"{'mode':<8} {'hit@1':>6} {f'hit@{k}':>6} {f'MRR@{k}':>6}"]
    for mode in modes:
        m = metrics(ranks[mode], k)
        out.append(f"{mode:<8} {m.hit_at_1:>6.3f} {m.hit_at_k:>6.3f} {m.mrr:>6.3f}")
    out.append("")
    out.append("first relevant rank per query (- = miss within top k)")
    out.append("  ".join(f"{m:>7}" for m in modes) + "  query")
    for i, case in enumerate(cases):
        cells = "  ".join(f"{ranks[m][i] if ranks[m][i] is not None else '-':>7}" for m in modes)
        out.append(f"{cells}  {case.query[:90]}")
    missed = [c.query for i, c in enumerate(cases) if all(ranks[m][i] is None for m in modes)]
    out.append("")
    out.append(f"missed by every mode: {len(missed)}")
    out.extend(f"  - {q}" for q in missed)
    if any(c.synthetic for c in cases):
        out.extend(["", SYNTHETIC_NOTE])
    return "\n".join(out)


def synthetic_query(chunk: Chunk) -> str | None:
    if chunk.heading:
        return chunk.heading
    candidates = [
        line.strip() for line in chunk.text.split("\n")
        if len(re.findall(r"[A-Za-z]{3,}", line)) >= 4
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda line: len(set(re.findall(r"[A-Za-z]{3,}", line.lower()))))[:160]


def generate(store: Store, n: int, seed: int) -> list[Case]:
    """Sample chunks and turn each into a query. Text found in several files is skipped,
    since only the sampled file counts as relevant and the rest would score as misses."""
    ids = store.all_chunk_ids()
    random.Random(seed).shuffle(ids)
    cases: list[Case] = []
    for start in range(0, len(ids), 200):
        batch = ids[start : start + 200]
        found = store.chunks(batch)
        for chunk in (found[cid] for cid in batch):
            query = synthetic_query(chunk)
            if query and len(cases) < n and store.paths_containing(query) == 1:
                cases.append(Case(query, (chunk.path,), synthetic=True))
        if len(cases) >= n:
            break
    return cases


def write_cases(cases: list[Case], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps({"query": c.query, "expected": list(c.expected), "synthetic": c.synthetic}) + "\n")
