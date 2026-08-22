# Run this on your own machine

You do **not** need a GPU, and you do **not** need to transcribe anything.
The 126 transcripts (68.6 hours of lecture) are already in this repo — the
expensive part is done.

Setup is three commands and one free API key.

---

## 1. Install

```bash
git clone <REPO_URL>
cd ytscraper
uv sync
```

No `uv`? `pip install uv` first, or use plain `pip install -e .` in a venv.

## 2. One API key

Copy the example env file:

```bash
cp .env.example .env
```

Open `.env` and paste in **one** of these (both free, no card needed):

- **Groq** — https://console.groq.com/keys → `GROQ_API_KEY=`
- **Gemini** — https://aistudio.google.com/apikey → `GEMINI_API_KEY=`

That's the only required setup. You don't need a Qdrant account — the index
is stored in a local folder by default.

## 3. Build the index

```bash
uv run ytrag reindex
```

This reads the transcripts in `transcripts/`, chunks them, embeds them, and
builds the search index. First run downloads the embedding model.

- **~5 minutes** and ~2.2GB download on the default model
- On a low-RAM machine, use the small model instead:
  `YTRAG_EMBED_MODEL=all-MiniLM-L6-v2 uv run ytrag reindex`

## 4. Use it

```bash
uv run ytrag serve
```

Open http://127.0.0.1:8000 — ask a question, click a timestamp, the lecture
plays from that exact second.

Or from the terminal:

```bash
uv run ytrag ask "memoization aur tabulation ka difference"
uv run ytrag search "sliding window"     # timestamps only, no LLM
uv run ytrag stats
```

---

## Every time after that

The setup above is **once**. The index is saved to disk, so from then on it is
one command:

```bash
uv run ytrag serve
```

Takes about **15 seconds** to start — that is the embedding model loading, and
it prints `Ready:` when it is done. After that every search is ~1 second.

Close the terminal or press **Ctrl-C** and the site stops; `localhost:8000`
goes dead until you start it again. Nothing is lost — the index stays on disk.
You never need to re-run `reindex` unless you change the embedding model or
add new lectures.

Prefer the terminal? These need no server at all:

```bash
uv run ytrag search "sliding window"    # timestamps only, instant, no LLM
uv run ytrag ask "kadane ka intuition"  # with a written answer
```

> **One at a time.** With the default local index only one process can use it
> at once, so stop `ytrag serve` before running `ytrag ask` in another
> terminal. (Setting `QDRANT_URL` to a hosted Qdrant lifts that restriction.)

---

## If something breaks

```bash
uv run ytrag preflight
```

Checks every dependency — keys, vector store, embedding model, LLM — and tells
you exactly which one is unhappy.

| symptom | fix |
|---|---|
| `GROQ_API_KEY is not set` | your `.env` is missing or in the wrong folder |
| `No transcripts` on reindex | run it from the project root, where `transcripts/` is |
| Out of memory | use `YTRAG_EMBED_MODEL=all-MiniLM-L6-v2` |
| LLM quota errors | free tiers are limited; `ytrag search` needs no LLM at all |

---

## Point it at your own videos

```bash
uv run ytrag langtest "<one video URL>"           # pick the Whisper language
uv run ytrag ingest --playlist "<PLAYLIST_URL>"   # download, transcribe, index
```

This is the part that wants a GPU — roughly 8x realtime, so an hour of video
takes about seven minutes. See the README for the details.
