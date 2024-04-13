import logging
import os

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

    task = TaskApi("ownapi")
    data = task.task()

    task.answer("https://kraina.hipopotamowie.pl/api/answer")
