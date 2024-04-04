import logging
import os

from dotenv import load_dotenv, find_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool

from libs.TaskApi import TaskApi


def addUser(name: str, surname: str, year: int) -> int:
    """Add user to database.

    Args:
        name: User name
        surname: User surname
        year: Year of birth
    """
    return 2


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("functions")
    data = task.task()

    defin = convert_to_openai_tool(addUser)
    task.answer(defin["function"])
