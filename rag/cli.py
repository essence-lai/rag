import argparse
import os
import sys
from pathlib import Path

from rag.answer import answer
from rag.embeddings import Embedder, FastEmbedEmbedder, HashEmbedder
from rag.evaluate import evaluate, generate, load_cases, report, write_cases
from rag.indexer import index
from rag.models import RagError, citation
from rag.search import MODES, search
from rag.store import Store

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_DIR / ".rag" / "index.db"
EMBEDDERS = {"fastembed": FastEmbedEmbedder, "hash": HashEmbedder}


def make_embedder(name: str) -> Embedder:
    return EMBEDDERS[name]()


def open_store(args: argparse.Namespace, embedder: Embedder | None) -> Store:
    path = Path(args.index)
    if args.command != "index" and not path.is_file():
        raise RagError(f"no index at {path}; run `rag index <dir>` first")
    return Store(path, embedder)


def cmd_index(args: argparse.Namespace) -> None:
    root = Path(args.dir).expanduser()
    if not root.is_dir():
        raise RagError(f"not a directory: {root}")
    embedder = make_embedder(args.embedder)
    store = open_store(args, embedder)
    r = index(root, store, embedder, exclude=(PROJECT_DIR, Path(args.index).resolve().parent))
    print(f"indexed {root.resolve()}")
    print(f"added {r.added}  updated {r.updated}  removed {r.removed}  unchanged {r.unchanged}  chunks {r.chunks}")


def cmd_search(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    store = open_store(args, embedder)
    for n, hit in enumerate(search(args.query, store, embedder, k=args.k, mode=args.mode), 1):
        ranks = f"dense#{hit.dense_rank or '-'} keyword#{hit.keyword_rank or '-'}"
        heading = f"  ({hit.chunk.heading})" if hit.chunk.heading else ""
        print(f"{citation(n, hit.chunk)}  score={hit.score:.4f}  {ranks}{heading}")
        snippet = " ".join(hit.chunk.text.split())
        print(f"    {snippet[:200]}")


def cmd_ask(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    store = open_store(args, embedder)
    print(answer(args.query, search(args.query, store, embedder, k=args.k)))


def cmd_eval(args: argparse.Namespace) -> None:
    cases = load_cases(Path(args.file))
    embedder = make_embedder(args.embedder)
    store = open_store(args, embedder)
    print(f"{len(cases)} queries, k={args.k}, index root {store.meta('root')}")
    print(report(cases, evaluate(cases, store, embedder, args.k), args.k))


def cmd_eval_generate(args: argparse.Namespace) -> None:
    store = open_store(args, None)
    cases = generate(store, args.n, args.seed)
    write_cases(cases, Path(args.out))
    print(f"wrote {len(cases)} synthetic queries to {args.out} (seed {args.seed}); "
          "these are easier than real questions")


def cmd_stats(args: argparse.Namespace) -> None:
    store = open_store(args, None)
    files, chunks = store.counts()
    print(f"index     {store.path}")
    print(f"root      {store.meta('root')}")
    print(f"embedder  {store.meta('embedder')} (dim {store.meta('dim')})")
    print(f"files     {files}")
    print(f"chunks    {chunks}")
    print(f"size      {os.path.getsize(store.path) / 1e6:.1f} MB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Local RAG over a directory.")
    parser.add_argument("--index", default=os.environ.get("RAG_INDEX", str(DEFAULT_INDEX)),
                        help="SQLite index file (default: %(default)s)")
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), default=os.environ.get("RAG_EMBEDDER", "fastembed"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="index or re-index a directory")
    p.add_argument("dir")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="retrieve chunks")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer with citations")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="retrieval eval over a JSONL golden set")
    p.add_argument("file")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("eval-generate", help="write a synthetic eval set sampled from the index")
    p.add_argument("out")
    p.add_argument("-n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_eval_generate)

    p = sub.add_parser("stats", help="show index contents")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    if getattr(args, "k", 1) < 1 or getattr(args, "n", 1) < 1:
        parser.error("-k and -n must be positive")
    try:
        args.func(args)
    except RagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
