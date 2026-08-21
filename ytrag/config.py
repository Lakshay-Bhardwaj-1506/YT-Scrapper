"""All tunables live here, every one of them env-driven.

Nothing else in the package reads os.getenv directly.
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# ------------------------------------------------------------------
# .env loading
# ------------------------------------------------------------------
# Keys (GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY) live in the repo-root .env,
# same as every other week. Look in the cwd chain first, then fall back to
# walking up from this file so `ytrag` works from any directory.
load_dotenv(find_dotenv(usecwd=True))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROOT_ENV = _REPO_ROOT / ".env"
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV, override=False)


# ------------------------------------------------------------------
# Runtime data directories
# ------------------------------------------------------------------
# These live OUTSIDE the repo. The split matters:
#   audio/       deletable, regenerable in seconds
#   transcripts/ precious, hours of GPU time, never delete
# Everything downstream of transcripts is cheap to rebuild, which is why
# changing your mind about the embedding model later costs nothing.
ROOT = Path(os.getenv("YTRAG_ROOT", Path.home() / ".ytrag"))
AUDIO_DIR = ROOT / "audio"
TRANSCRIPT_DIR = ROOT / "transcripts"

for _d in (AUDIO_DIR, TRANSCRIPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Transcription
# ------------------------------------------------------------------
# faster-whisper (CTranslate2), not mlx-whisper — this is a Windows/CUDA box.
WHISPER_MODEL = os.getenv("YTRAG_WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("YTRAG_WHISPER_DEVICE", "auto")  # auto | cuda | cpu
WHISPER_COMPUTE = os.getenv("YTRAG_WHISPER_COMPUTE", "")  # blank = pick per device
WHISPER_LANG = os.getenv("YTRAG_WHISPER_LANG", "en")  # see README §6.0 — TEST THIS FIRST
WHISPER_BEAM = int(os.getenv("YTRAG_WHISPER_BEAM", 5))
# Batched inference transcribes several VAD-detected speech regions at once.
# Same model, same weights, several times the throughput — the difference
# between an overnight run and an afternoon one on 68 hours of lecture.
# Set to 0 to fall back to the sequential path.
WHISPER_BATCH = int(os.getenv("YTRAG_WHISPER_BATCH", 8))


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------
# 75s is roughly one explained idea (~180-220 words of speech).
CHUNK_SECONDS = int(os.getenv("YTRAG_CHUNK_SECONDS", 75))
CHUNK_OVERLAP_SECONDS = int(os.getenv("YTRAG_CHUNK_OVERLAP", 15))
MIN_CHUNK_WORDS = int(os.getenv("YTRAG_MIN_CHUNK_WORDS", 15))
# Retrieval hits the chunk containing the answer, but the explanation usually
# starts just before it. Rewind the citation link by a few seconds.
LINK_REWIND_SECONDS = int(os.getenv("YTRAG_LINK_REWIND", 5))


# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------
# bge-m3 is multilingual and handles code-switched Hinglish, which is the
# whole retrieval problem here. It is ~2.2GB — for a small deploy tier set
# YTRAG_EMBED_MODEL=all-MiniLM-L6-v2 and run `ytrag reindex`. Cheap, because
# transcripts are cached: minutes of embedding, not hours of transcription.
EMBED_MODEL = os.getenv("YTRAG_EMBED_MODEL", "BAAI/bge-m3")
EMBED_BATCH = int(os.getenv("YTRAG_EMBED_BATCH", 16))
# Some models want an instruction prefixed to the *query only*. bge-m3 does
# not; bge-*-en-v1.5 does. If you switch models, check the model card.
EMBED_QUERY_PREFIX = os.getenv("YTRAG_EMBED_QUERY_PREFIX", "")


# ------------------------------------------------------------------
# Vector store (Qdrant, same as day14 / day15)
# ------------------------------------------------------------------
# YouTube starts challenging bulk downloaders after a few dozen videos
# ("Sign in to confirm you're not a bot"). A short random pause between
# downloads keeps us under that radar; it costs minutes across a whole
# playlist and avoids losing the run to a rate limit.
DOWNLOAD_SLEEP_MIN = float(os.getenv("YTRAG_DOWNLOAD_SLEEP_MIN", 2))
DOWNLOAD_SLEEP_MAX = float(os.getenv("YTRAG_DOWNLOAD_SLEEP_MAX", 6))
# If you get challenged persistently, send YouTube cookies.
#
# YTRAG_COOKIES_FROM_BROWSER works for Firefox, but NOT for Chrome or Edge on
# Windows: since Chrome 127 they encrypt cookies with App-Bound Encryption and
# yt-dlp cannot decrypt them (yt-dlp issue #10927). For Chromium browsers use
# YTRAG_COOKIES_FILE instead — export cookies.txt with a "Get cookies.txt"
# browser extension and point this at the file.
COOKIES_FROM_BROWSER = os.getenv("YTRAG_COOKIES_FROM_BROWSER", "")
COOKIES_FILE = os.getenv("YTRAG_COOKIES_FILE", "")


QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
# The collection name carries the embedding dim, so switching embedding
# models can never hit a dimension-mismatch error against a live collection,
# and a local index and a deploy index can coexist in the same cluster.
COLLECTION = os.getenv("YTRAG_COLLECTION", "dsa_lectures")
UPSERT_BATCH = int(os.getenv("YTRAG_UPSERT_BATCH", 128))


# ------------------------------------------------------------------
# Retrieval + answering
# ------------------------------------------------------------------
TOP_K = int(os.getenv("YTRAG_TOP_K", 6))
# Cosine distance = 1 - score. Anything above this is treated as "not really
# about the question" and dropped before the LLM ever sees it.
#
# 0.5, not the 0.75 the plan suggested. bge-m3 compresses its distances hard:
# measured on Hinglish DSA text, on-topic queries land around 0.35-0.47 and
# clearly off-topic ones ("React hooks", "capital of France") around 0.52-0.71.
# A 0.75 cutoff never fires at all, which quietly disables the entire guard.
#
# This is still a starting point, not an answer. On a full index an off-topic
# query has thousands more chances to find one spuriously close chunk, so the
# off-topic floor drifts down. Re-tune against the golden set once the real
# playlist is ingested — `ytrag search` prints the raw distances.
MAX_DISTANCE = float(os.getenv("YTRAG_MAX_DISTANCE", 0.5))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("YTRAG_LLM_MODEL", "openai/gpt-oss-120b")

# The exact string the system says when retrieval comes back empty. Kept here
# because evaluate.py and the frontend both need to recognise it.
REFUSAL = "Ye topic in lectures me cover nahi hua."


# ------------------------------------------------------------------
# API (Phase 3/4)
# ------------------------------------------------------------------
MAX_QUESTION_CHARS = int(os.getenv("YTRAG_MAX_QUESTION_CHARS", 500))
RATE_LIMIT_REQUESTS = int(os.getenv("YTRAG_RATE_LIMIT_REQUESTS", 20))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("YTRAG_RATE_LIMIT_WINDOW", 60))
