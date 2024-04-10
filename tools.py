import json
import logging
import os
import pprint

import requests
from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers.openai_tools import (
    PydanticToolsParser,
    JsonOutputToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.tools import tool

from libs.TaskApi import TaskApi


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Assign user message to one of the tools and nothing more. Always return JSON object only.

tools###
- ToDo: user would like to remember something and message does not include any kind of date to accomplish this
- Calendar: User would like to do something at specific date

examples###
{examples}

Current date: {date}""",
            ),
            ("human", "{question}"),
        ]
    )
    examples = """Przypomnij mi, że mam kupić mleko
{"tool":"ToDo","desc":"Kup mleko" }
Jutro mam spotkanie z Marianem
{"tool":"Calendar","desc":"Spotkanie z Marianem","date":"2024-04-11"}
Przygotowac prezentacja na środę"""

    task = TaskApi("tools")
    question = task.task()["question"]

    llm = OpenAI()
    chain = LLMChain(llm=llm, prompt=prompt)
    ret = chain.invoke(dict(examples=examples, date="2024-04-11", question=question))
    print(ret["text"].strip())
    task.answer(json.loads(ret["text"].strip()))
