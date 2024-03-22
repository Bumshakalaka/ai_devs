import logging

from libs.TaskApi import TaskApi
from libs.openaiApi import OpenApi

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    openai = OpenApi()
    task = TaskApi("moderation")
    data = task.task()
    ret = []
    for input in data["input"]:
        ret.append(openai.moderations(input))
        logger.info(f"{input}: {ret[-1]}")
    assert task.answer(ret)
