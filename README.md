# YT Lecture RAG — AI-Powered Search & Q&A over YouTube Lectures

> Ask any question in English and get a **grounded answer + clickable timestamps** that jump to the exact second it was explained in the lecture.

```
"What is the difference between memoization and tabulation?"

→ Answer: 3–4 lines, built only from what was actually said
→ Sources:
   [1] DP Lecture 3 — Memoization Deep Dive        @ 12:04   ▸ jump
   [2] DP Lecture 5 — Tabulation Recipe            @ 04:31   ▸ jump
```

Most RAG demos return a blob of text. This one returns **a place in a video**.

---

## What it does

- **Transcribes** a YouTube playlist using OpenAI Whisper (`large-v3`, batched for speed)
- **Chunks** transcripts in the time domain, preserving timestamps — not by character count
- **Embeds** each chunk with `BAAI/bge-m3` (1024-dim) and upserts into Qdrant
- **Answers** questions with a Groq LLM grounded strictly on retrieved chunks — refusing to hallucinate if context is weak
- **Serves** everything through a FastAPI backend + a single-page UI with an embedded YouTube player

---

## Architecture

```
YouTube Playlist
      │
      ▼
  yt-dlp  ──►  m4a audio
                   │
                   ▼
          faster-whisper (large-v3, batch=8)
                   │
                   ▼
           transcripts/*.json   ◄──── cached, never re-transcribed
                   │
                   ▼
          chunk.py  (time-window, 75s + 15s overlap)
                   │
                   ▼
      sentence-transformers (BAAI/bge-m3)
                   │
                   ▼
           Qdrant Cloud (vector store)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  /search endpoint        /ask endpoint
  (semantic retrieval,    (retrieval + Groq LLM,
   no LLM, instant)       grounded answering)
        │                     │
        └──────────┬──────────┘
                   ▼
          FastAPI + Single-Page UI
          (embedded YouTube IFrame player)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Playlist download | `yt-dlp` |
| Transcription | `faster-whisper` — `large-v3`, batched (8.1× real-time on RTX GPU) |
| Chunking | Custom time-domain chunker (preserves segment timestamps) |
| Embeddings | `sentence-transformers` + `BAAI/bge-m3` (1024-dim) |
| Vector store | **Qdrant Cloud** |
| LLM | **Groq** (`openai/gpt-oss-120b`) |
| CLI | `typer` + `rich` |
| API | `FastAPI` with rate limiting, per-IP, question-length capped |
| UI | Single HTML page — YouTube IFrame API, glassmorphism design |

---

## Key Design Decisions

### Time-domain chunking, not text splitting
Generic splitters (`RecursiveCharacterTextSplitter`) concatenate text and lose timestamps. The timestamp *is* the product here — `chunk.py` operates on the Whisper segment list directly, in the time domain, and never splits a segment.

### Grounding-first answer pipeline
Three guards prevent hallucination:
1. Chunks above `MAX_DISTANCE` (0.5 for bge-m3) are dropped before the LLM is called
2. If nothing survives the distance filter, the system refuses — no LLM call, no fake answer
3. An answer with zero matched citations is flagged `grounded: false` and shown without links

### Cached transcripts as the source of truth
Transcription (`large-v3`, batched) runs at **8.1× real-time** on an RTX GPU — but it still takes hours for a full playlist. The transcript JSON files are written atomically (`tempfile → os.replace`) so a crash mid-write never corrupts the cache. Re-running the ingest skips any already-transcribed video — chunk sizes, embedding models, and vector stores can all change without re-transcribing.

### Idempotent ingest
Point IDs are `uuid5("{video_id}:{start_sec}")` — deterministic, stable across re-runs. Re-indexing is always an upsert, never a duplicate.

---

## Project Structure

```
ytscraper/
├── api/
│   ├── main.py              # FastAPI app — /ask, /search, /stats, /meta
│   └── static/
│       └── index.html       # Single-page UI
├── ytrag/
│   ├── config.py            # All env-driven configuration
│   ├── models.py            # Video, Segment, Chunk dataclasses
│   ├── transcribe.py        # Whisper transcription + atomic caching
│   ├── chunk.py             # Time-window chunker + hallucination filter
│   ├── embed.py             # Sentence-transformer embedder
│   ├── index.py             # Qdrant upsert + search
│   ├── answer.py            # Retrieval + grounded LLM answering
│   ├── playlist.py          # yt-dlp playlist fetching
│   ├── evaluate.py          # Golden-set evaluation harness
│   └── cli.py               # typer CLI (ingest / ask / search / eval / serve …)
├── eval/
│   └── golden.json          # Retrieval + refusal test cases
├── pyproject.toml
└── .env.example
```

---

## Setup

**Prerequisites:** Python 3.11, `uv` package manager

```bash
# Clone and enter the project
git clone <repo-url>
cd ytscraper

# Create virtual environment and install dependencies
uv venv --python 3.11
uv sync

# For CUDA-accelerated transcription (recommended)
uv sync --extra cuda
```

**Environment variables** — copy `.env.example` to `.env` and fill in:

```env
GROQ_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

Free tiers work fine: [Groq Console](https://console.groq.com) · [Qdrant Cloud](https://cloud.qdrant.io)

---

## Usage

### 1. Preflight check
```bash
uv run ytrag preflight --playlist "<PLAYLIST_URL>"
```
Validates all keys, Qdrant connectivity, the embedding model, Whisper, and the LLM before any long run.

### 2. Ingest a playlist
```bash
# Test with one video first
uv run ytrag ingest --playlist "<PLAYLIST_URL>" --limit 1

# Full playlist
uv run ytrag ingest --playlist "<PLAYLIST_URL>"
```
Per video: download audio → transcribe (cached) → delete audio → chunk → embed → upsert.

The run is fully resumable — re-running picks up where it left off. One failing video is logged and skipped; the rest continue.

### 3. Search & ask from the CLI
```bash
uv run ytrag search "binary search on answer"          # semantic retrieval, shows distances
uv run ytrag ask "difference between BFS and DFS"      # retrieval + LLM answer
```

### 4. Run the web UI
```bash
uv run ytrag serve
# or
uvicorn api.main:app --reload
```

Open **http://127.0.0.1:8000** — click any result to play the lecture at that exact second.

### Other commands
```bash
uv run ytrag reindex              # re-embed from cached transcripts (e.g. after changing embed model)
uv run ytrag reindex --replace    # clear old chunks first, then reindex
uv run ytrag eval --verbose       # run golden-set evaluation
uv run ytrag stats                # index stats
uv run ytrag clean-audio          # delete downloaded audio files
```

---

## Configuration

All settings are environment-driven. Defaults in [`ytrag/config.py`](ytrag/config.py):

| Variable | Default | Description |
|---|---|---|
| `YTRAG_WHISPER_MODEL` | `large-v3` | Whisper model size |
| `YTRAG_WHISPER_DEVICE` | `auto` | `cuda` / `cpu` to force |
| `YTRAG_WHISPER_BATCH` | `8` | Batch size — `0` to disable |
| `YTRAG_WHISPER_BEAM` | `5` | Beam search width |
| `YTRAG_WHISPER_LANG` | `en` | Transcription language |
| `YTRAG_CHUNK_SECONDS` | `75` | Chunk window (≈ one explained idea) |
| `YTRAG_CHUNK_OVERLAP` | `15` | Overlap between consecutive chunks |
| `YTRAG_EMBED_MODEL` | `BAAI/bge-m3` | Embedding model |
| `YTRAG_COLLECTION` | `dsa_lectures` | Qdrant collection name |
| `YTRAG_TOP_K` | `6` | Results returned per query |
| `YTRAG_MAX_DISTANCE` | `0.5` | Grounding cutoff (bge-m3 compressed distances) |
| `YTRAG_LLM_MODEL` | `openai/gpt-oss-120b` | Groq model |
| `YTRAG_RATE_LIMIT_REQUESTS` | `10` | Max requests per window (per IP) |

---

## Evaluation

The golden set at [`eval/golden.json`](eval/golden.json) contains two types of test cases:

**Retrieval hit** — passes if the expected video appears in top-k *and* a returned chunk overlaps the expected timestamp within tolerance:
```json
{
  "q": "difference between memoization and tabulation",
  "expect_video_id": "abc123",
  "expect_around_sec": 724,
  "tolerance_sec": 120
}
```

**Refusal** — passes if the pipeline declines to answer an out-of-scope question:
```json
{
  "q": "how do React hooks work",
  "expect_refusal": true
}
```

```bash
uv run ytrag eval --verbose
```

---

## Transcription Speed

Measured on an RTX GPU:

| Config | Speed | Notes |
|---|---|---|
| `large-v3`, sequential, beam 5 | 2.5× real-time | Baseline |
| `large-v3`, **batch 8**, beam 5 | **8.1×** | Default — same quality |
| `large-v3`, batch 8, beam 1 | 10.5× | Faster but greedy decoding |

Batching is not a quality trade-off — same model, same weights. Sequential inference leaves the GPU idle between speech regions. Batched inference fills it.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | — | Serves the single-page UI |
| `GET /health` | — | LLM + embed model info |
| `GET /stats` | — | Index stats (chunk count, etc.) |
| `GET /meta` | — | Lecture count + hours indexed |
| `POST /search` | `{question, top_k}` | Semantic retrieval, no LLM |
| `POST /ask` | `{question, top_k}` | Retrieval + grounded LLM answer |

---

## License

MIT
