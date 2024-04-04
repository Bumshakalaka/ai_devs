import logging
import os

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings

from libs.TaskApi import TaskApi

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.INFO
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("embedding")
    task.task()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    text_to_process = "Hawaiian pizza"
    query_result = embeddings.embed_query(text_to_process)
    assert task.answer(query_result)
