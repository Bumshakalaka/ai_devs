import logging
import os
import re
import tempfile

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from libs.TaskApi import TaskApi
import requests


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.INFO
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("whisper")
    data = task.task()
    if ret := re.match(r".+(?P<url>https://.+mp3)", data["msg"]):
        url = ret.group("url")
    client = OpenAI()
    resp = requests.get(url, stream=True)
    with open("/tmp/aa.mp3", "wb") as fd:
        fd.write(resp.content)
    audio_file = open("/tmp/aa.mp3", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    task.answer(transcription.text)
