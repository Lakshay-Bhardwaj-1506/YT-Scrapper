"""Retrieve -> grounded answer + citations.

The important part of this module is what it does when retrieval comes back
empty: it returns the refusal without calling the LLM at all.

The model knows DSA perfectly well. If it is handed junk context it will
happily answer from its own training and attach your timestamps to it — the
student clicks the link and you are talking about something else entirely.
That is strictly worse than saying "cover nahi hua".
"""

import re

from groq import Groq

from ytrag.config import GROQ_API_KEY, LLM_MODEL, MAX_DISTANCE, REFUSAL, TOP_K
from ytrag.index import search
from ytrag.models import Chunk

_CLIENT: Groq | None = None

SYSTEM_PROMPT = f"""You are answering using ONLY the transcript excerpts below, which come from
Pratyush's DSA lectures. The transcripts are auto-generated and may contain
minor errors — read past obvious mis-transcriptions of technical terms.

Rules:
- Answer only from the excerpts. If they don't cover it, say exactly:
  "{REFUSAL}"
- Cite with [1], [2] inline, using the excerpt numbers given below.
- Match the language of the question (Hinglish question -> Hinglish answer).
- 4-6 sentences max.
- Never invent a timestamp or a lecture name."""

_CITATION_RE = re.compile(r"\[(\d+)\]")


def get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to the repo-root .env.")
        _CLIENT = Groq(api_key=GROQ_API_KEY)
    return _CLIENT


def build_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f'[{i}] "{chunk.video_title}" @ {chunk.timestamp}\n{chunk.text}')
    return "\n\n".join(blocks)


def _citation(chunk: Chunk, distance: float) -> dict:
    return {
        "title": chunk.video_title,
        "timestamp": chunk.timestamp,
        "url": chunk.url,
        "start_sec": chunk.link_sec,
        "video_id": chunk.video_id,
        "distance": round(distance, 4),
    }


def _renumber(text: str, hits: list[tuple[Chunk, float]]) -> tuple[str, list[dict]]:
    """Keep only the citations the model actually used, and renumber them 1..N.

    Without this the student sees six links under an answer that only used
    one, and stops trusting any of them.
    """
    order: list[int] = []
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if 1 <= idx <= len(hits) and idx not in order:
            order.append(idx)

    if not order:
        return text, []

    remap = {old: new for new, old in enumerate(order, start=1)}
    rewritten = _CITATION_RE.sub(
        lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else "",
        text,
    )
    citations = [_citation(*hits[old - 1]) for old in order]
    return rewritten, citations


def answer(
    question: str,
    top_k: int = TOP_K,
    video_id: str | None = None,
    max_distance: float | None = None,
) -> dict:
    """-> {"answer", "citations", "grounded", "retrieved"}"""
    question = question.strip()
    if not question:
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": 0}

    hits = search(question, top_k=top_k, video_id=video_id, max_distance=max_distance)

    # Guard one: nothing survived the distance cutoff, so there is nothing to
    # ground an answer in. Return the refusal and never call the LLM.
    if not hits:
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": 0}

    chunks = [chunk for chunk, _ in hits]
    user_prompt = f"EXCERPTS\n{build_context(chunks)}\n\nQUESTION: {question}"

    response = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    text = (response.choices[0].message.content or "").strip()

    # Guard two: the model read the excerpts and said they don't cover it.
    if REFUSAL.lower() in text.lower():
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": len(hits)}

    text, citations = _renumber(text, hits)

    # Guard three: an answer with no citation at all is the model talking from
    # its own knowledge. Show it, but don't dress it up with links.
    return {
        "answer": text,
        "citations": citations,
        "grounded": bool(citations),
        "retrieved": len(hits),
    }


def retrieve_only(question: str, top_k: int = TOP_K, filtered: bool = False) -> list[tuple[Chunk, float]]:
    """Retrieval without the LLM — used by evaluate.py and `ytrag search`.

    Unfiltered by default: 2.0 is the maximum possible cosine distance, so
    nothing is dropped. Eval wants to see what retrieval actually returned,
    including the results the MAX_DISTANCE cutoff would have thrown away.
    """
    return search(question, top_k=top_k, max_distance=None if filtered else 2.0)
