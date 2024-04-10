import logging
import os

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain import globals

from libs.TaskApi import TaskApi

logger = logging.getLogger(__name__)


def analyse_image(url: str) -> str:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(temperature=0.5, model="gpt-4-vision-preview", max_tokens=1024)
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "I will give you a drawing of a gnome with a hat on his head. "
                        "Tell me what is the color of the hat in POLISH. "
                        "If any errors occur, return 'ERROR' as answer",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                ]
            )
        ]
    )
    return msg.content


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("gnome")
    url = task.task()["url"]
    msg = analyse_image(url)
    print(msg)
    task.answer(msg)
