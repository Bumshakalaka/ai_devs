import logging
import os

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from libs.TaskApi import TaskApi


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    COLLECTION_NAME = "unknownNews"

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("search")
    query = task.task()["question"]

    qdrant = QdrantClient(host="localhost", port=6333)
    embeddings = OpenAIEmbeddings()
    query_embed = embeddings.embed_query(query)
    search = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embed,
        limit=1,
    )
    print(search)
    task.answer(search[0].payload["url"])
