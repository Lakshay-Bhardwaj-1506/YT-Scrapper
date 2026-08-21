"""FastAPI wrapper around answer(), plus the single-page UI.

Phase 4 asks for a rate limit and a question-length cap before this goes
public — both are here, because "add it later" never happens.
"""

import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ytrag import config
from ytrag.answer import answer as answer_question
from ytrag.index import stats as index_stats

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="YT Lecture RAG", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_CHARS)
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


# A dict of deques is enough for one process on a free tier. Behind more than
# one worker this becomes per-worker, so move it to Redis before it matters.
_HITS: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _HITS[client]

    while window and now - window[0] > config.RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()

    if len(window) >= config.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Slow down — max {config.RATE_LIMIT_REQUESTS} questions per "
            f"{config.RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    window.append(now)


@app.get("/health")
def health():
    return {"status": "ok", "model": config.LLM_MODEL, "embed_model": config.EMBED_MODEL}


@app.get("/stats")
def stats():
    info = index_stats()
    info.pop("videos", None)
    return info


@app.post("/ask")
def ask(payload: AskRequest, request: Request):
    _rate_limit(request)
    try:
        return answer_question(payload.question, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
