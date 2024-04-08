import json
import logging
import os
from pathlib import Path

import hnswlib
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def SearchVectorStore(query, k):
    p = hnswlib.Index(space="cosine", dim=1536)
    p.load_index(str(Path(__file__).parent / "memory/memory.index"), max_elements=100)
    with open(Path(__file__).parent / "memory/docstore.json", "r") as fd:
        docs = json.load(fd)
    new_embedding = OpenAIEmbeddings(model="text-embedding-ada-002").embed_query(query)
    labels, dist = p.knn_query(new_embedding, k=k)
    return [x[1] for x in docs["content"] if x[0] in labels]


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)
    query = "How big is Rajesh apartment"
    context = "\n".join(SearchVectorStore(query, 2))
    chat = ChatOpenAI(model="gpt-4", temperature=1.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                f"""
                Answer questions as truthfully using the context below and nothing more. If you don't know the answer, say "don't know".
                context###${context}###
                """
            ),
            HumanMessage(content=query),
        ]
    )
    chain = prompt | chat
    ret = chain.invoke({})
    print(ret)
