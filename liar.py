import json
import logging
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

from libs.TaskApi import TaskApi

system_prompt = """
Validate question by returning only words YES or NO and nothing else.
YES - if the answer is correct
NO - if the answer is not correct

Example:
Q: What is the capital city of Poland?
A: The film's dinosaurs were brought to life through a combination of animatronics, puppetry, and CGI.
Return: NO

Q: What is the capital city of Poland?
A: Capital city of Poland is Warsaw
Return: YES

"""

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.INFO
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("liar")
    task.task()
    question = "Jaka jest stolica Polski"
    data = task.task_question(question)
    task_answer = data["answer"]
    llm = OpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Q:{question}\nA:{task_answer}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    ret = chain.invoke(
        {
            "question": question,
            "task_answer": task_answer,
        }
    )
    resp = ret["text"].replace("\n", "").replace("Return:", "").strip()
    assert task.answer(resp)
