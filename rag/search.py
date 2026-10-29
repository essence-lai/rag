from typing import Literal

import numpy as np

from rag.embeddings import Embedder
from rag.models import Hit
from rag.store import Store

Mode = Literal["hybrid", "dense", "keyword"]
MODES: tuple[Mode, ...] = ("dense", "keyword", "hybrid")
CANDIDATES = 50
RRF_K = 60


def dense_ranking(query: str, store: Store, embedder: Embedder, limit: int) -> list[tuple[str, float]]:
    ids, matrix = store.matrix()
    if not ids:
        return []
    scores = matrix @ embedder.embed([query])[0]
    top = np.argsort(-scores, kind="stable")[:limit]
    return [(ids[i], float(scores[i])) for i in top]


def search(query: str, store: Store, embedder: Embedder, k: int = 5, mode: Mode = "hybrid") -> list[Hit]:
    dense = dense_ranking(query, store, embedder, max(k, CANDIDATES)) if mode != "keyword" else []
    keyword = store.keyword(query, max(k, CANDIDATES)) if mode != "dense" else []
    dense_rank = {cid: r for r, (cid, _) in enumerate(dense, 1)}
    keyword_rank = {cid: r for r, cid in enumerate(keyword, 1)}

    if mode == "dense":
        scored = dense
    elif mode == "keyword":
        scored = [(cid, 1.0 / r) for cid, r in keyword_rank.items()]
    else:
        fused: dict[str, float] = {}
        for ranks in (dense_rank, keyword_rank):
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r)
        scored = sorted(fused.items(), key=lambda item: -item[1])

    top = scored[:k]
    chunks = store.chunks([cid for cid, _ in top])
    return [Hit(chunks[cid], score, dense_rank.get(cid), keyword_rank.get(cid)) for cid, score in top]
