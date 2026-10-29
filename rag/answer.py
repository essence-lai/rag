import importlib.util
import os

from rag.models import Hit, RagError, citation

DEFAULT_MODEL = "claude-sonnet-5-5"
# Models that accept the server-side `fallbacks: "default"` form on the Claude API.
# Sending it to any other model is a 400, so an overridden RAG_MODEL goes without.
FALLBACK_MODELS = frozenset({"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"})
SNIPPET_CHARS = 240

SYSTEM = (
    "You answer questions using only the numbered sources provided. "
    "Cite every claim with the source number in square brackets, like [1] or [2][3]. "
    "If the sources do not contain the answer, say so plainly and do not guess. "
    "Keep the answer short."
)


def citation_list(hits: list[Hit], snippets: bool) -> str:
    lines = []
    for n, hit in enumerate(hits, 1):
        lines.append(citation(n, hit.chunk))
        if snippets:
            snippet = " ".join(hit.chunk.text.split())
            lines.append("    " + (snippet[:SNIPPET_CHARS] + "..." if len(snippet) > SNIPPET_CHARS else snippet))
    return "\n".join(lines)


def sources_prompt(query: str, hits: list[Hit]) -> str:
    blocks = [f"{citation(n, h.chunk)}\n{h.chunk.text}" for n, h in enumerate(hits, 1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}"


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and importlib.util.find_spec("anthropic") is not None


def answer(query: str, hits: list[Hit]) -> str:
    if not hits:
        return "No sources matched the question."
    if not llm_available():
        return citation_list(hits, snippets=True)
    return _ask_claude(query, hits) + "\n\nSources:\n" + citation_list(hits, snippets=False)


def _ask_claude(query: str, hits: list[Hit]) -> str:
    import anthropic

    model = os.environ.get("RAG_MODEL", DEFAULT_MODEL)
    extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if model in FALLBACK_MODELS else {}
    try:
        response = anthropic.Anthropic().beta.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": sources_prompt(query, hits)}],
            **extra,
        )
    except anthropic.APIStatusError as exc:
        raise RagError(f"Claude API returned {exc.status_code} for model {model}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise RagError(f"could not reach the Claude API: {exc}") from exc
    if response.stop_reason == "refusal":
        return "The model declined to answer this question."
    return "".join(block.text for block in response.content if block.type == "text").strip()
