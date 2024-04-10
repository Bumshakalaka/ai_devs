import logging
import os
import pprint

import requests
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from libs.TaskApi import TaskApi


@tool
def other(question: str, answer: str) -> str:
    """
    Answer briefly for user question.
    """
    return answer


@tool
def currency(currency_code: str) -> str:
    """Exchange rate for currency code (3-digits Code for specific currency, f.e USD, PLN, EUR)."""
    url = f"http://api.nbp.pl/api/exchangerates/rates/A/{currency_code}"
    response = requests.get(url)
    return response.json()["rates"][0]["mid"]


@tool
def population(country_name="POLAND"):
    """Population based on country name (country name is in English, e.g. Poland, Germany)."""
    url = f"https://restcountries.com/v3.1/name/{country_name}"
    response = requests.get(url)
    return response.json()[0]["population"]


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("knowledge")
    question = task.task()["question"]
    # question = "kto napisał Romeo i Julię?"
    # question = "Ile ludzi mieszka w Polsce?"
    # question = "jak nazywa się stolica Czech?"
    print(question)

    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", verbose=True)
    llm_with_tools = llm.bind_tools([other, currency, population])
    tool_chain = llm_with_tools | JsonOutputToolsParser()
    ret = tool_chain.invoke(question)
    pprint.pprint(ret)
    func = ret[0]["type"]
    aa = locals()[func].invoke(ret[0]["args"])
    print(aa)
    task.answer(aa)
