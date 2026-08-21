"""Qdrant upsert + query.

Same client as day14/day15, but with two differences that matter at this scale:
the collection name carries the embedding dim, and point IDs are derived from
a stable chunk_id so re-ingesting overwrites instead of duplicating.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ytrag.config import (
    COLLECTION,
    MAX_DISTANCE,
    QDRANT_API_KEY,
    QDRANT_URL,
    TOP_K,
    UPSERT_BATCH,
)
from ytrag.embed import get_embedder
from ytrag.models import Chunk
from ytrag.util import with_retry

_CLIENT: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _CLIENT
    if _CLIENT is None:
        if not QDRANT_URL:
            raise RuntimeError("QDRANT_URL is not set. Add it to the repo-root .env.")
        _CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    return _CLIENT


def collection_name() -> str:
    """e.g. 'dsa_lectures_1024'.

    Qdrant rejects vectors whose size does not match the collection, so
    stamping the dim into the name means switching embedding models creates a
    new collection instead of erroring — and lets a 1024-dim local index and a
    384-dim deploy index live side by side.
    """
    return f"{COLLECTION}_{get_embedder().dim}"


def ensure_collection() -> str:
    """Create the collection if it does not exist. Safe to call every time."""
    client = get_client()
    name = collection_name()

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=get_embedder().dim,
                distance=Distance.COSINE,
            ),
        )
        # Needed for the --video filter on search.
        client.create_payload_index(
            collection_name=name,
            field_name="video_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    return name


def upsert_chunks(chunks: list[Chunk], batch_size: int = UPSERT_BATCH) -> int:
    """Embed and upsert. Idempotent: same chunk_id -> same point ID -> overwrite."""
    if not chunks:
        return 0

    name = ensure_collection()
    client = get_client()
    embedder = get_embedder()

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        points = [
            PointStruct(id=c.point_id, vector=v, payload=c.to_payload())
            for c, v in zip(batch, vectors)
        ]
        # Retried for the same reason downloads are: this is a network call in
        # the middle of a long unattended run. Upsert is idempotent, so a
        # retry after a partial success is harmless.
        with_retry(
            lambda: client.upsert(collection_name=name, points=points, wait=True),
            label=f"upsert {len(points)} points",
        )
        total += len(points)

    return total


def delete_video(video_id: str) -> None:
    """Remove every chunk for one video. Used when re-chunking with new settings."""
    client = get_client()
    name = ensure_collection()
    client.delete(
        collection_name=name,
        points_selector=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        ),
        wait=True,
    )


def indexed_video_ids() -> set[str]:
    """Which videos already have chunks in the collection.

    Lets a re-run skip the embed+upsert for work already done. Without this,
    restarting a partly-finished ingest re-embeds every cached transcript
    before reaching new material — minutes of idle GPU each time.
    """
    client = get_client()
    name = collection_name()
    if not client.collection_exists(name):
        return set()

    found: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=["video_id"],
            with_vectors=False,
        )
        for point in points:
            if point.payload and point.payload.get("video_id"):
                found.add(point.payload["video_id"])
        if offset is None:
            break
    return found


def search(
    query: str,
    top_k: int = TOP_K,
    video_id: str | None = None,
    max_distance: float | None = None,
) -> list[tuple[Chunk, float]]:
    """Return [(chunk, distance)] sorted best-first, already distance-filtered.

    Qdrant returns a cosine *similarity* score (higher is better); the rest of
    the system thinks in distance (lower is better), so convert once here.
    """
    name = ensure_collection()
    client = get_client()
    vector = get_embedder().embed_query(query)

    query_filter = None
    if video_id:
        query_filter = Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        )

    results = client.query_points(
        collection_name=name,
        query=vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    ).points

    cutoff = MAX_DISTANCE if max_distance is None else max_distance
    hits: list[tuple[Chunk, float]] = []
    for point in results:
        distance = 1.0 - float(point.score)
        if distance > cutoff:
            continue
        hits.append((Chunk.from_payload(point.payload), distance))

    return hits


def stats() -> dict:
    """Collection size plus a per-video breakdown."""
    client = get_client()
    name = collection_name()

    if not client.collection_exists(name):
        return {"collection": name, "exists": False, "chunks": 0, "videos": {}}

    info = client.get_collection(name)
    videos: dict[str, dict] = {}

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=512,
            offset=offset,
            with_payload=["video_id", "video_title"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            vid = payload.get("video_id", "?")
            entry = videos.setdefault(vid, {"title": payload.get("video_title", "?"), "chunks": 0})
            entry["chunks"] += 1
        if offset is None:
            break

    return {
        "collection": name,
        "exists": True,
        "chunks": info.points_count or 0,
        "dim": get_embedder().dim,
        "embed_model": get_embedder().name,
        "videos": videos,
    }
