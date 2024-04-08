import json
import os
from pathlib import Path

import hnswlib
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings


def create_vector_db(docs: Path, out: Path) -> None:
    with open(docs, "r") as fd:
        docs = json.load(fd)
    content = [x[1] for x in docs["content"]]

    # gen embedings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002").embed_documents(
        content
    )

    # Create a new index
    p = hnswlib.Index(space="cosine", dim=len(embeddings[0]))

    # Initialize an index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=100, ef_construction=100, M=16)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    p.set_ef(10)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    p.set_num_threads(4)

    # # Element insertion (can be called several times)
    p.add_items(embeddings)
    p.save_index(str(out))
    del p


def vector_db_search(index: Path, docs: Path, query, k):
    p = hnswlib.Index(
        space="cosine", dim=1536
    )  # dim -> OpenAIEmbeddings(model="text-embedding-ada-002")
    p.load_index(str(index), max_elements=100)
    with open(docs, "r") as fd:
        docs = json.load(fd)

    query_embed = OpenAIEmbeddings(model="text-embedding-ada-002").embed_query(query)
    labels, dist = p.knn_query(query_embed, k=k)
    return [x[1] for x in docs["content"] if x[0] in labels]


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    create_vector_db(
        Path(__file__).parent / "memory/docstore.json",
        Path(__file__).parent / "memory/memory.index",
    )
    ret = vector_db_search(
        Path(__file__).parent / "memory/memory.index",
        Path(__file__).parent / "memory/docstore.json",
        "Rajesh apartment",
        2,
    )
    print(ret)
