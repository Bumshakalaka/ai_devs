import os
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_core.documents import Document


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    COLLECTION_NAME = "unknownNews"

    qdrant = QdrantClient(host="localhost", port=6333)
    embeddings = OpenAIEmbeddings()
    query = "Do"
    query_embed = embeddings.embed_query(query)
    result = qdrant.get_collections()
    indexed = True if COLLECTION_NAME in result.collections else False
    if not indexed:
        qdrant.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1536, distance=Distance.COSINE, on_disk=True
            ),
        )
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        news = requests.get("https://unknow.news/archiwum_aidevs.json").json()
        points = []
        for nn in news:
            points.append(
                dict(
                    id=str(uuid.uuid4()),
                    payload=nn,
                    vector=embeddings.embed_query(nn["info"]),
                )
            )
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )
