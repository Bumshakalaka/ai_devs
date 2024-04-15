import logging
import os

from dotenv import load_dotenv, find_dotenv

from libs.TaskApi import TaskApi
from libs.renderform import RenderFormApi

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("meme")
    data = task.task()
    api = RenderFormApi()
    pic_data = {
        "TEXT.text": data["text"],
        "PIC.src": data["image"],
    }
    meme_url = api.render("sedate-wasps-dive-warmly-1825", pic_data)

    task.answer(meme_url)
