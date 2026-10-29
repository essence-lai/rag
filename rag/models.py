from dataclasses import dataclass


class RagError(Exception):
    """A user-facing error raised at a boundary (bad path, bad file, wrong embedder)."""


@dataclass(frozen=True)
class Document:
    path: str
    text: str
    sha256: str


@dataclass(frozen=True)
class Chunk:
    id: str
    path: str
    start_line: int
    end_line: int
    heading: str | None
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    keyword_rank: int | None


def chunk_id(path: str, start_line: int, end_line: int) -> str:
    return f"{path}:{start_line}-{end_line}"


def citation(n: int, chunk: Chunk) -> str:
    return f"[{n}] {chunk.path}:L{chunk.start_line}-{chunk.end_line}"
