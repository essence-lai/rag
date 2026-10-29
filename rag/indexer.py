from dataclasses import dataclass
from pathlib import Path

from rag.chunking import chunk
from rag.embeddings import Embedder
from rag.loader import load
from rag.models import Chunk
from rag.store import Store


@dataclass(frozen=True)
class IndexReport:
    added: int
    updated: int
    removed: int
    unchanged: int
    chunks: int


def embedding_text(c: Chunk) -> str:
    # The path and heading carry context ("which project is this?") that the chunk body often lacks.
    return "\n".join(part for part in (c.path, c.heading, c.text) if part)


def index(root: Path, store: Store, embedder: Embedder, exclude: tuple[Path, ...] = ()) -> IndexReport:
    """Bring the store in line with root. Safe to rerun after a crash: each file commits alone."""
    known = store.file_hashes()
    seen: set[str] = set()
    added = updated = unchanged = 0
    with store.db:
        store.set_meta("root", str(root.resolve()))
    for doc in load(root, exclude):
        seen.add(doc.path)
        if known.get(doc.path) == doc.sha256:
            unchanged += 1
            continue
        chunks = chunk(doc)
        store.replace_file(doc.path, doc.sha256, chunks, embedder.embed([embedding_text(c) for c in chunks]))
        if doc.path in known:
            updated += 1
        else:
            added += 1
    removed = known.keys() - seen
    for path in sorted(removed):
        store.remove_file(path)
    return IndexReport(added, updated, len(removed), unchanged, store.counts()[1])
