import logging
import os

import requests
from dotenv import load_dotenv, find_dotenv

from libs.TaskApi import TaskApi
from langsmith import Client

client = Client()

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("google")
    data = task.task()
    task.answer("https://kraina.hipopotamowie.pl/api/search")
