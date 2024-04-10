import os
import uuid
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, ScoredPoint
from langchain_core.documents import Document

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    qdrant = QdrantClient(host="localhost", port=6333)
    embeddings = OpenAIEmbeddings()
    query = "Do you know the name of Adam's dog?"
    query_embed = embeddings.embed_query(query)
    result = qdrant.get_collections()
    indexed = True if result.collections else False
    if not indexed:
        qdrant.create_collection(
            "ai_devs",
            vectors_config=VectorParams(
                size=1536, distance=Distance.COSINE, on_disk=True
            ),
        )
    collection_info = qdrant.get_collection("ai_devs")
    if collection_info.points_count == 0:
        memory = TextLoader(Path(__file__).parent / "memory/memory.md").load()
        points = []
        for chunk in memory[0].page_content.split("\n\n"):
            d = Document(page_content=chunk)
            metadata = {
                "source": "ai_devs",
                "content": d.page_content,
                "uuid": str(uuid.uuid4()),
            }
            points.append(
                dict(
                    id=str(uuid.uuid4()),
                    payload=metadata,
                    vector=embeddings.embed_query(d.page_content),
                )
            )
        qdrant.upsert(
            collection_name="ai_devs",
            points=[
                PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )
    search = qdrant.search(
        collection_name="ai_devs",
        query_vector=query_embed,
        limit=1,
    )
    print(search)
