# rag

`rag` is a small retrieval-augmented search tool for a local directory. It indexes text and code files into one SQLite file and runs hybrid search (embeddings plus BM25). Every result cites a file and an exact line range. It can draft an answer with Claude when an API key is set, and it includes a retrieval evaluator.

It is a CLI and a library. It uses no framework, only numpy, fastembed, and the standard library.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'        # add ,llm for answer generation: '.[llm,dev]'
.venv/bin/pytest -q
```

The first `rag index` downloads the quantized `BAAI/bge-small-en-v1.5` ONNX model (64 MB on disk) to fastembed's cache, which defaults to the system temp directory. Set `FASTEMBED_CACHE_PATH` to keep it somewhere the OS will not clean. After that, everything runs offline unless you use `rag ask` with an API key.

## Quickstart

```sh
rag index ~/notes                      # build or update .rag/index.db
rag search "how do I roll back a deploy" -k 5
rag search "AirPlay port 5000" --mode keyword
rag ask "why does the service avoid port 5000?"
rag stats
```

The index stores the root it was built from, so `search`, `ask`, and `eval` do not need the directory again.

## How it works

**Loading** (`rag/loader.py`). The loader walks the root in sorted order. It skips `.git`, `node_modules`, `.venv`, `venv`, `__pycache__`, `.pytest_cache`, `dist`, `build`, `.next`, `.rag`, this project's own checkout, and the index directory. It reads only allowlisted extensions (md, txt, rst, py, js, ts, tsx, jsx, json, yaml, yml, toml, sh, html, css, sql). It skips files over 1 MB, lockfiles, and files that are not valid UTF-8. A `.ragignore` at the root holds gitignore-style globs, matched with `fnmatch` against the relative path and the file name. Line endings are normalised to `\n`, and the SHA-256 of the raw bytes identifies the file version.

**Chunking** (`rag/chunking.py`). A registry maps each extension to a chunker. Markdown splits on headings outside code fences, and each chunk carries its heading path, such as `Deploy guide > Rollback`. Code finds the starts of top-level blocks: a line after a blank line, or a `def`, `class`, `function`, `export`, or `const` at column 0. Other text splits on paragraphs. All chunkers share one packer. It fills a window up to 1200 characters, cuts at the last block start that fits, and starts the next window 2 lines back for overlap. It never splits a line. Blank edges are trimmed and whitespace-only windows are dropped.

**Citations.** A chunk's id is `path:start-end` with 1-based inclusive lines, and `"\n".join(lines[start-1:end])` equals the chunk text exactly. The tests check this on every chunk. Results render as `[1] git-meta/docs/RUNBOOK.md:L18-19`.

**Embeddings** (`rag/embeddings.py`). `Embedder` is a protocol with `name`, `dim`, and `embed(texts)`, which returns L2-normalised float32 rows. `FastEmbedEmbedder` runs bge-small-en-v1.5 locally (384 dimensions). `HashEmbedder` is a deterministic hashing-trick bag of words, used by the tests and available with `--embedder hash`. The text embedded for each chunk is its path, heading, and body, because the body alone often lacks the project name.

**Indexing** (`rag/indexer.py`, `rag/store.py`). The index is one SQLite file with these tables:

- `meta` holds the embedder name, the dimension, and the root.
- `files` holds the path and SHA-256 of each file.
- `chunks` holds each chunk's lines, text, and vector blob.
- `chunks_fts` is an FTS5 table over path, heading, and text.

Opening an index with a different embedder raises an error that names both embedders. Each run compares hashes. Only new or changed files are re-chunked and re-embedded, and chunks of deleted files are removed. Each file is swapped in its own transaction. A crash leaves every file either fully old or fully new, and the next run converges. A second run with no changes reports all files unchanged.

**Search** (`rag/search.py`). `dense` is brute-force cosine similarity over the vector matrix. `keyword` is FTS5 BM25. Every query word is quoted and the words are joined with OR, so punctuation and FTS operators in the query cannot raise errors. `hybrid`, the default, applies reciprocal rank fusion with k=60 to the dense top 50 and the keyword top 50. Each `Hit` keeps both ranks so you can see which retriever found it.

**Answering** (`rag/answer.py`). If `ANTHROPIC_API_KEY` is set and `anthropic` is installed, `rag ask` sends the numbered chunks to Claude. The default model is `claude-sonnet-5-5`, and `RAG_MODEL` overrides it. The prompt tells the model to cite `[n]` and to say when the sources do not contain the answer. The output is the answer followed by the citation list. The server-side refusal fallback is enabled for models that support it. Without a key, `rag ask` prints the citations with snippets.

## Evaluating retrieval

Eval sets describe the indexed corpus, so `evals/*.jsonl` is gitignored and none ship with this repo. A golden set is JSONL. Each line is `{"query": "...", "expected": ["path", "dir/prefix/"]}`. A hit counts as relevant when its chunk path equals an expected entry or starts with it.

```sh
rag eval evals/my-questions.jsonl -k 5
rag eval-generate evals/synthetic.jsonl -n 50 --seed 7
rag eval evals/synthetic.jsonl
```

For each mode (dense, keyword, hybrid), `rag eval` reports these metrics:

- **hit@1** is the fraction of queries whose first result is relevant.
- **hit@k** (recall@k) is the fraction with any relevant result in the top k.
- **MRR@k** is the mean of 1/rank of the first relevant result, counting 0 when none is in the top k.

It also prints the first relevant rank per query for each mode and lists the queries that every mode missed.

**Writing a golden set.** Read the documents first, then write questions the way a user would ask them. Do not copy lines from the documents. List every file that answers the question. Use a directory prefix when several files in a project answer it. Keep questions that name the project and questions that do not, because the path is part of the indexed text and naming the project makes a question easier. Twenty questions is enough to catch large regressions but not small ones. One query moves hit@1 by 0.05 on a 20-question set.

**Synthetic sets.** `rag eval-generate` samples chunks with a seed. It uses the chunk's heading path as the query, or else its most distinctive line, and expects that chunk's file. It skips text that appears in more than one file, because only the sampled file counts as relevant. These queries copy the indexed text, so they are easier than real questions and favour keyword search. Use them to catch regressions, not to estimate quality.

**Comparing modes and chunk sizes.** All three modes run on every eval, so a mode comparison is one command. To compare chunk sizes, change `MAX_CHARS` or `OVERLAP_LINES` in `rag/chunking.py`, build a separate index with `--index /tmp/rag-800.db index <dir>`, and run the same eval file against each index with `--index`. Compare the same golden set at the same k.

### Measured results

These results are for the author's local projects folder, measured on 2026-10-29. The corpus is 374 files and 1607 chunks, embedded with fastembed bge-small-en-v1.5, with k=5.

Hand-written set (22 questions, not published):

| mode    | hit@1 | hit@5 | MRR@5 |
|---------|-------|-------|-------|
| dense   | 0.773 | 1.000 | 0.864 |
| keyword | 0.818 | 1.000 | 0.898 |
| hybrid  | 0.909 | 1.000 | 0.941 |

Synthetic set (50 queries, seed 7, not published). These queries are easier than real questions:

| mode    | hit@1 | hit@5 | MRR@5 |
|---------|-------|-------|-------|
| dense   | 0.620 | 0.760 | 0.677 |
| keyword | 0.940 | 0.980 | 0.957 |
| hybrid  | 0.680 | 0.860 | 0.761 |

Read these numbers with the following caveats:

- hit@5 is 1.0 in every mode on the hand-written set, so that set cannot separate the modes at k=5. Only hit@1 and MRR separate them. One of the two hybrid misses at rank 1 is the rollback question, because the git-meta and sdn runbooks contain the same rollback text.
- By a rough count, 44 of the 50 synthetic queries are lines of code, CSS, or JSON rather than headings. For exact code strings the dense ranking is noisy and dilutes keyword search's rank-1 match under fusion, so hybrid scores below keyword there.
- Across seeds 1, 2, and 7, synthetic hybrid hit@1 was 0.88, 0.92, and 0.68, so a single synthetic run is noisy.

## Configuration

| setting | how |
|---|---|
| index file | `--index PATH` or `RAG_INDEX` (default `<project>/.rag/index.db`) |
| embedder | `--embedder fastembed\|hash` or `RAG_EMBEDDER` (default `fastembed`) |
| answer model | `RAG_MODEL` (default `claude-sonnet-5-5`) |
| LLM answers | `ANTHROPIC_API_KEY` set and `pip install -e '.[llm]'` |
| exclusions | `.ragignore` in the indexed root |
| chunk size, overlap | `MAX_CHARS`, `OVERLAP_LINES` in `rag/chunking.py` |

## Limitations

- Dense search is brute force over an in-memory matrix. That is fine for tens of thousands of chunks, not millions.
- A single line longer than 1200 characters, such as minified JavaScript, becomes one oversized chunk. Built bundles outside `dist`/`build` get indexed, so exclude them with `.ragignore`.
- The extension allowlist leaves out Go, Ruby, PHP, and other languages, so those files are not indexed.
- The loader always skips file names that look like credentials (`*secret*`, `*.env`, `.env*`, `*credential*`, `*.pem`, `*.key`, `id_rsa*`). A secret stored under any other name is indexed, so add it to `.ragignore` before indexing, especially before using `rag ask`, which sends chunks to an API.
- Changing the embedder requires a new index file. The store refuses a mismatched embedder and does not migrate.
- bge-small is an English model with a 512-token input limit. Chunks are capped at 1200 characters, well under that limit.
