from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]):
        """Upload vectors with their payloads to the collection."""
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def delete_by_source(self, source_id: str):
        """
        Delete all vectors for a given source_id.
        Called by the TTL cleanup job after 10 minutes.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_id),
                    )
                ]
            ),
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_filters: list[str] | None = None,
    ) -> dict:
        """
        Search for chunks similar to the query vector.

        source_filters is REQUIRED. Passing None is a hard error — this prevents
        accidental cross-tenant reads if the caller forgets to supply an ID.

        Uses a Qdrant `should` (OR) filter so callers can search across multiple
        documents belonging to the same session in a single round-trip.
        """
        if not source_filters:
            raise ValueError(
                "source_filters must be a non-empty list. "
                "Unfiltered search is not permitted in a multi-tenant context."
            )

        query_filter = Filter(
            should=[
                FieldCondition(key="source", match=MatchValue(value=sid))
                for sid in source_filters
            ]
        )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=top_k,
        )

        contexts: list[str] = []
        sources: set[str] = set()

        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            # Use the human-readable filename stored at ingest time.
            # Fall back to source_id only if filename is missing.
            display_name = payload.get("filename") or payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(display_name)

        return {"contexts": contexts, "sources": list(sources)}
