import re
import sqlite3
from pathlib import Path

import numpy as np

from rag.embeddings import Embedder
from rag.models import Chunk, RagError

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    heading TEXT,
    text TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, path, heading, text);
"""


def fts_query(query: str) -> str:
    """Quote every word so FTS5 operators and punctuation in user input are inert."""
    return " OR ".join(f'"{token}"' for token in re.findall(r"\w+", query))


class Store:
    """SQLite index: chunk rows with vectors, plus an FTS5 table for BM25."""

    def __init__(self, path: Path, embedder: Embedder | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self._matrix: tuple[list[str], np.ndarray] | None = None
        if embedder is not None:
            self._bind(embedder)

    def _bind(self, embedder: Embedder) -> None:
        name, dim = self.meta("embedder"), self.meta("dim")
        if name is None:
            with self.db:
                self.set_meta("embedder", embedder.name)
                self.set_meta("dim", str(embedder.dim))
        elif (name, int(dim or 0)) != (embedder.name, embedder.dim):
            raise RagError(
                f"index {self.path} was built with embedder {name} (dim {dim}), "
                f"but {embedder.name} (dim {embedder.dim}) was requested; "
                "use the same embedder or a different --index"
            )

    def close(self) -> None:
        self.db.close()

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def file_hashes(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT path, sha256 FROM files"))

    def replace_file(self, path: str, sha256: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Swap a file's chunks in one transaction, so a crash leaves old or new, never half."""
        with self.db:
            self._delete_chunks(path)
            self.db.executemany(
                "INSERT INTO chunks (id, path, start_line, end_line, heading, text, vector) VALUES (?,?,?,?,?,?,?)",
                [
                    (c.id, c.path, c.start_line, c.end_line, c.heading, c.text, v.astype(np.float32).tobytes())
                    for c, v in zip(chunks, vectors)
                ],
            )
            self.db.executemany(
                "INSERT INTO chunks_fts (id, path, heading, text) VALUES (?,?,?,?)",
                [(c.id, c.path, c.heading or "", c.text) for c in chunks],
            )
            self.db.execute("INSERT OR REPLACE INTO files (path, sha256) VALUES (?, ?)", (path, sha256))
        self._matrix = None

    def remove_file(self, path: str) -> None:
        with self.db:
            self._delete_chunks(path)
            self.db.execute("DELETE FROM files WHERE path = ?", (path,))
        self._matrix = None

    def _delete_chunks(self, path: str) -> None:
        self.db.execute("DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE path = ?)", (path,))
        self.db.execute("DELETE FROM chunks WHERE path = ?", (path,))

    def matrix(self) -> tuple[list[str], np.ndarray]:
        if self._matrix is None:
            rows = self.db.execute("SELECT id, vector FROM chunks ORDER BY id").fetchall()
            dim = int(self.meta("dim") or 0)
            vectors = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), dim)
            self._matrix = ([r[0] for r in rows], vectors)
        return self._matrix

    def keyword(self, query: str, limit: int) -> list[str]:
        match = fts_query(query)
        if not match:
            return []
        rows = self.db.execute(
            "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (match, limit)
        )
        return [r[0] for r in rows]

    def chunks(self, ids: list[str]) -> dict[str, Chunk]:
        found: dict[str, Chunk] = {}
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            rows = self.db.execute(
                f"SELECT id, path, start_line, end_line, heading, text FROM chunks WHERE id IN ({','.join('?' * len(batch))})",
                batch,
            )
            found.update({r[0]: Chunk(*r) for r in rows})
        return found

    def chunks_for(self, path: str) -> list[Chunk]:
        rows = self.db.execute(
            "SELECT id, path, start_line, end_line, heading, text FROM chunks WHERE path = ? ORDER BY start_line", (path,)
        )
        return [Chunk(*r) for r in rows]

    def paths_containing(self, text: str) -> int:
        return self.db.execute("SELECT COUNT(DISTINCT path) FROM chunks WHERE instr(text, ?) > 0", (text,)).fetchone()[0]

    def all_chunk_ids(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT id FROM chunks ORDER BY id")]

    def counts(self) -> tuple[int, int]:
        files = self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return files, chunks
