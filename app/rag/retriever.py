import uuid

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient

from app.config import QdrantSettings

ChunkKey = tuple[str, int, int]  # (book, chapter, chunk)


# Must stay the same formula book-parser uses when it stores the chunks.
def point_id(book: str, chapter: int, chunk: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{book}/{chapter}/{chunk}"))


class PassageRetriever(BaseRetriever):
    """Finds the top-k chunks, widens each with its neighbours and merges adjacent ones."""

    client: QdrantClient
    collection: str
    embeddings: Embeddings
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        hits = self.client.query_points(
            self.collection, query=vector, limit=self.k, with_payload=True
        ).points

        # Chunks are short, so each hit is widened with the chunk before and after.
        # A set, because neighbouring hits share chunks.
        keys: set[ChunkKey] = set()
        # Only the hits have a score; the neighbours fetched by id don't.
        scores: dict[ChunkKey, float] = {}
        for hit in hits:
            payload = hit.payload or {}
            book, chapter, chunk = payload["book"], payload["chapter"], payload["chunk"]
            scores[(book, chapter, chunk)] = hit.score
            for neighbour in range(max(chunk - 1, 0), chunk + 2):
                keys.add((book, chapter, neighbour))

        records = self.client.retrieve(
            self.collection, ids=[point_id(*key) for key in keys], with_payload=True
        )
        payloads = sorted(
            (record.payload for record in records if record.payload),
            key=lambda payload: (payload["book"], payload["chapter"], payload["chunk"]),
        )

        passages: list[Document] = []
        previous: ChunkKey | None = None
        for payload in payloads:
            book, chapter, chunk = payload["book"], payload["chapter"], payload["chunk"]
            score = scores.get((book, chapter, chunk), 0.0)
            if previous == (book, chapter, chunk - 1):
                passages[-1].page_content += " " + payload["text"]
                passages[-1].metadata["score"] = max(passages[-1].metadata["score"], score)
            else:
                passages.append(
                    Document(
                        page_content=payload["text"],
                        metadata={"book": book, "chapter": chapter, "score": score},
                    )
                )
            previous = (book, chapter, chunk)
        return sorted(passages, key=lambda doc: doc.metadata["score"], reverse=True)


def create_retriever(settings: QdrantSettings, embeddings: Embeddings, k: int) -> BaseRetriever:
    client = QdrantClient(
        url=settings.url,
        api_key=settings.api_key.get_secret_value() if settings.api_key else None,
    )
    return PassageRetriever(
        client=client, collection=settings.collection, embeddings=embeddings, k=k
    )
