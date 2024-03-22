import logging

from libs.TaskApi import TaskApi

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("helloapi")
    data = task.task()
    assert task.answer(data["cookie"])
