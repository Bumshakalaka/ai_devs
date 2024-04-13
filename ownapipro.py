import logging
import os

import requests
from dotenv import load_dotenv, find_dotenv

from libs.TaskApi import TaskApi


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("ownapipro")
    data = task.task()
    ret = requests.get("https://kraina.hipopotamowie.pl/api/newchat")
    print(ret.text)
    task.answer("https://kraina.hipopotamowie.pl/api/chat")
