import logging
import os

import requests
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from libs.TaskApi import TaskApi


def md2html(input: str):
    system_prompt = """Convert md document to html and nothing more.
    Special rules overwrites default converter rules:
    - **bold** - <span class=\"bold\">bold text</span>
    - _underline_ - <u>underline</u>
    examples:
    **bold text**
    <span class=\"bold\">bold text</span>
    
    regular text\n**bold text**
    <p>regular text</>\n<span class=\"bold\">bold text</span>
    
    _podkreślenie_
    <u>podkreślenie</u>
    
    regular text\n**bold text**\n_something_\n*italic*
    <p>regular text</>\n<span class=\"bold\">bold text</span>\n<u>something</u>\n<em>italic</em>   
    
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(temperature=1.0, model="gpt-4-turbo")
    chain = prompt | llm
    ret = chain.invoke(dict(input=input))
    return ret.content


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("md2html")
    data = task.task()
    ret = md2html(data["input"])
    logger.info(ret)
    task.answer(ret)
