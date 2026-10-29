"""Split documents into line-addressed chunks.

Each chunker decides where a file's natural units begin (sections, paragraphs,
top-level code blocks). `_pack` then turns those units into windows of at most
MAX_CHARS, preferring to cut at a unit boundary and never splitting a line.
"""

import re
from collections.abc import Callable
from pathlib import PurePosixPath

from rag.models import Chunk, Document, chunk_id

MAX_CHARS = 1200
OVERLAP_LINES = 2

Chunker = Callable[[Document], list[Chunk]]

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_CODE_START = re.compile(
    r"^(async\s+def|def|class|function|export|const|let|var|interface|type|enum|"
    r"module\.exports|describe|it|test|CREATE|SELECT|INSERT|ALTER)\b"
)


def _size(lines: list[str], start: int, end: int) -> int:
    return sum(len(line) + 1 for line in lines[start:end])


def _pack(
    path: str, lines: list[str], start: int, end: int, boundaries: set[int], heading: str | None
) -> list[Chunk]:
    """Window lines[start:end] (0-based, half-open) into chunks."""
    chunks: list[Chunk] = []
    lo = start
    while lo < end:
        hi = lo + 1
        while hi < end and _size(lines, lo, hi + 1) <= MAX_CHARS:
            hi += 1
        cut = hi
        if hi < end:
            preferred = [b for b in boundaries if lo < b <= hi]
            if preferred:
                cut = max(preferred)
        chunks.extend(_emit(path, lines, lo, cut, heading))
        if cut >= end:
            break
        nxt = max(cut - OVERLAP_LINES, lo + 1)
        if _size(lines, nxt, cut + 1) > MAX_CHARS:
            nxt = cut
        lo = nxt
    return chunks


def _emit(path: str, lines: list[str], lo: int, hi: int, heading: str | None) -> list[Chunk]:
    while lo < hi and not lines[lo].strip():
        lo += 1
    while hi > lo and not lines[hi - 1].strip():
        hi -= 1
    if lo == hi:
        return []
    start_line, end_line = lo + 1, hi
    return [Chunk(chunk_id(path, start_line, end_line), path, start_line, end_line, heading, "\n".join(lines[lo:hi]))]


def _paragraph_starts(lines: list[str], start: int, end: int) -> set[int]:
    return {i for i in range(start + 1, end) if lines[i].strip() and not lines[i - 1].strip()}


def chunk_markdown(doc: Document) -> list[Chunk]:
    lines = doc.text.split("\n")
    sections: list[tuple[int, str | None]] = [(0, None)]
    stack: list[tuple[int, str]] = []
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            level = len(match.group(1))
            stack = [(lvl, title) for lvl, title in stack if lvl < level] + [(level, match.group(2))]
            sections.append((i, " > ".join(title for _, title in stack)))
    chunks: list[Chunk] = []
    for (start, heading), (end, _) in zip(sections, sections[1:] + [(len(lines), None)]):
        chunks.extend(_pack(doc.path, lines, start, end, _paragraph_starts(lines, start, end), heading))
    return _dedupe(chunks)


def chunk_code(doc: Document) -> list[Chunk]:
    lines = doc.text.split("\n")
    boundaries = {
        i for i in range(1, len(lines))
        if lines[i].strip()
        and not lines[i][0].isspace()
        and (not lines[i - 1].strip() or (_CODE_START.match(lines[i]) and not lines[i - 1].startswith("@")))
    }
    return _dedupe(_pack(doc.path, lines, 0, len(lines), boundaries, None))


def chunk_text(doc: Document) -> list[Chunk]:
    lines = doc.text.split("\n")
    return _dedupe(_pack(doc.path, lines, 0, len(lines), _paragraph_starts(lines, 0, len(lines)), None))


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    return list({c.id: c for c in chunks}.values())


CHUNKERS: dict[str, Chunker] = {
    ".md": chunk_markdown,
    **dict.fromkeys(
        [".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".sql", ".css", ".html", ".json", ".yaml", ".yml", ".toml"],
        chunk_code,
    ),
}


def chunk(doc: Document) -> list[Chunk]:
    return CHUNKERS.get(PurePosixPath(doc.path).suffix.lower(), chunk_text)(doc)
