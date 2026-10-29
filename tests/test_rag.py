from pathlib import Path

import numpy as np
import pytest

from rag.chunking import MAX_CHARS, chunk
from rag.embeddings import HashEmbedder
from rag.evaluate import Case, evaluate, metrics
from rag.indexer import IndexReport, index
from rag.loader import load
from rag.models import Document, RagError
from rag.search import search
from rag.store import Store

GUIDE = """# Deploy guide

Intro paragraph about the service.

## Rollback

Revert the last commit on main.
Netlify redeploys automatically.

## Logs

Logs go to Splunk.
"""

CODE = "\n".join(
    [f"def handler_{i}(event):\n    value = event['amount'] * {i}\n    return value + {i}\n" for i in range(60)]
)

NOTES = "\n\n".join(f"Paragraph {i} " + "lorem ipsum dolor sit amet " * 6 for i in range(20))


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "GUIDE.md").write_text(GUIDE)
    (root / "app.py").write_text(CODE)
    (root / "notes.txt").write_text(NOTES)
    (root / "secret.md").write_text("zebra")
    (root / ".ragignore").write_text("secret.md\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("ignored()")
    (root / "binary.txt").write_bytes(b"\xff\xfe\x00bad")
    return root


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "index.db", HashEmbedder())


def test_loader_skips_ignored_dirs_files_and_bad_utf8(corpus: Path) -> None:
    assert sorted(d.path for d in load(corpus)) == ["app.py", "docs/GUIDE.md", "notes.txt"]


def test_loader_never_reads_secret_looking_files(tmp_path: Path) -> None:
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "local_secret.txt").write_text("password=hunter2")
    (tmp_path / ".env").write_text("KEY=1")
    (tmp_path / "notes.txt").write_text("hello")
    assert [d.path for d in load(tmp_path)] == ["notes.txt"]


def test_citation_line_ranges_reproduce_exact_text(corpus: Path) -> None:
    for doc in load(corpus):
        file_lines = (corpus / doc.path).read_text().split("\n")
        chunks = chunk(doc)
        assert chunks
        for c in chunks:
            assert "\n".join(file_lines[c.start_line - 1 : c.end_line]) == c.text
            assert c.id == f"{c.path}:{c.start_line}-{c.end_line}"
            assert c.text.strip()
            assert len(c.text) <= MAX_CHARS


def test_long_files_split_into_several_overlapping_windows(corpus: Path) -> None:
    doc = next(d for d in load(corpus) if d.path == "app.py")
    chunks = chunk(doc)
    assert len(chunks) > 3
    covered = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    non_blank = {i for i, line in enumerate(doc.text.split("\n"), 1) if line.strip()}
    assert non_blank <= covered
    assert any(b.start_line <= a.end_line for a, b in zip(chunks, chunks[1:]))


def test_markdown_heading_path_is_carried() -> None:
    chunks = chunk(Document("GUIDE.md", GUIDE, "x"))
    by_text = {c.text.split("\n")[-1]: c.heading for c in chunks}
    assert by_text["Netlify redeploys automatically."] == "Deploy guide > Rollback"
    assert by_text["Logs go to Splunk."] == "Deploy guide > Logs"


def test_markdown_ignores_hash_lines_inside_code_fences() -> None:
    text = "# Top\n\n```sh\n# not a heading\necho hi\n```\n"
    assert {c.heading for c in chunk(Document("a.md", text, "x"))} == {"Top"}


def test_reindex_twice_reports_all_unchanged(corpus: Path, store: Store) -> None:
    first = index(corpus, store, HashEmbedder())
    assert first.added == 3 and first.chunks > 0
    assert index(corpus, store, HashEmbedder()) == IndexReport(0, 0, 0, 3, first.chunks)


def test_edited_file_updates_only_its_chunks(corpus: Path, store: Store) -> None:
    index(corpus, store, HashEmbedder())
    untouched = {p: store.chunks_for(p) for p in ("app.py", "notes.txt")}
    (corpus / "docs" / "GUIDE.md").write_text(GUIDE.replace("Splunk", "Datadog"))
    report = index(corpus, store, HashEmbedder())
    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 1, 0, 2)
    assert {p: store.chunks_for(p) for p in untouched} == untouched
    guide = " ".join(c.text for c in store.chunks_for("docs/GUIDE.md"))
    assert "Datadog" in guide and "Splunk" not in guide
    assert store.keyword("Splunk", 10) == []


def test_deleted_file_chunks_are_gone(corpus: Path, store: Store) -> None:
    index(corpus, store, HashEmbedder())
    (corpus / "notes.txt").unlink()
    report = index(corpus, store, HashEmbedder())
    assert report.removed == 1
    assert store.chunks_for("notes.txt") == []
    assert all(not cid.startswith("notes.txt:") for cid in store.matrix()[0])
    assert store.keyword("lorem", 10) == []


def test_embedder_mismatch_raises_naming_both(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    Store(path, HashEmbedder(dim=512)).close()
    with pytest.raises(RagError, match="hash-512.*hash-64"):
        Store(path, HashEmbedder(dim=64))


@pytest.mark.parametrize("mode", ["keyword", "hybrid"])
def test_punctuation_in_query_does_not_raise(corpus: Path, store: Store, mode: str) -> None:
    index(corpus, store, HashEmbedder())
    for query in ['foo(bar)"', 'AND OR NOT *', '"', "-", "rollback: main^"]:
        search(query, store, HashEmbedder(), mode=mode)


class BlindEmbedder:
    """Maps every text to the same vector, so dense ranking carries no signal."""

    name = "blind"
    dim = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(texts), 1))


def test_hybrid_finds_file_by_exact_rare_token(corpus: Path, tmp_path: Path) -> None:
    (corpus / "notes.txt").write_text(NOTES + "\n\nThe magic id is QX7ZK91.\n")
    store = Store(tmp_path / "blind.db", BlindEmbedder())
    index(corpus, store, BlindEmbedder())
    hits = search("where is QX7ZK91", store, BlindEmbedder(), k=3, mode="hybrid")
    assert hits[0].chunk.path == "notes.txt"
    assert "QX7ZK91" in hits[0].chunk.text
    assert hits[0].keyword_rank == 1


def test_metrics_on_hand_computed_case() -> None:
    # ranks 1, 3, miss, 2 at k=3: hit@1 = 1/4, hit@3 = 3/4, MRR = (1 + 1/3 + 0 + 1/2) / 4
    m = metrics([1, 3, None, 2], k=3)
    assert m.hit_at_1 == pytest.approx(0.25)
    assert m.hit_at_k == pytest.approx(0.75)
    assert m.mrr == pytest.approx((1 + 1 / 3 + 0.5) / 4)
    assert metrics([5], k=3).hit_at_k == 0


def test_evaluate_counts_path_prefix_as_relevant(corpus: Path, store: Store) -> None:
    index(corpus, store, HashEmbedder())
    ranks = evaluate([Case("Netlify redeploys rollback revert", ("docs/",))], store, HashEmbedder(), k=3)
    assert ranks["keyword"] == [1]
