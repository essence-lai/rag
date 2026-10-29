import hashlib
import os
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path

from rag.models import Document

SKIP_DIRS = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build", ".next", ".rag"}
)
EXTENSIONS = frozenset(
    {".md", ".txt", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
     ".toml", ".sh", ".html", ".css", ".sql"}
)
LOCKFILES = frozenset({"package-lock.json", "yarn.lock", "pnpm-lock.yaml"})
# Indexed text can be sent to the Claude API by `rag ask`, so likely credentials never enter the index.
SECRET_PATTERNS = ("*secret*", "*.env", ".env*", "*credential*", "*.pem", "*.key", "id_rsa*")
MAX_BYTES = 1_000_000


def read_ignore(root: Path) -> list[str]:
    path = root / ".ragignore"
    if not path.is_file():
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def is_ignored(rel: str, patterns: list[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for raw in patterns:
        pattern = raw.strip("/")
        if fnmatch(rel, pattern) or fnmatch(name, pattern) or rel.startswith(pattern + "/"):
            return True
    return False


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load(root: Path, exclude: tuple[Path, ...] = ()) -> Iterator[Document]:
    """Yield every indexable file under root, in a stable order.

    `exclude` holds absolute directories to prune, such as this project's own
    checkout and index directory when they sit inside root.
    """
    root = root.resolve()
    excluded = {p.resolve() for p in exclude}
    patterns = [*SECRET_PATTERNS, *read_ignore(root)]
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS
            and (here / d) not in excluded
            and not is_ignored((here / d).relative_to(root).as_posix(), patterns)
        )
        for name in sorted(filenames):
            path = here / name
            rel = path.relative_to(root).as_posix()
            if path.suffix.lower() not in EXTENSIONS or name in LOCKFILES or is_ignored(rel, patterns):
                continue
            doc = _read(path, rel)
            if doc is not None:
                yield doc


def _read(path: Path, rel: str) -> Document | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            return None
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return Document(path=rel, text=normalize_newlines(text), sha256=hashlib.sha256(data).hexdigest())
