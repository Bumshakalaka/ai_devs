import json
import logging
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

from libs.TaskApi import TaskApi

system_prompt = """
As a food blogger specializing in Italian cuisine, particularly pizza, your task is to construct a recipe post utilizing various guide points provided. Take note that each paragraph should consist of no more than two sentences for better readability.

You'll be given the guide points as JSON objects, which are simply particular elements of your blog for you to develop. For instance:
```
{input}
```
From this, you might create a paragraph or two on pizza history, then another on essential ingredients and so on.

When you've finished, present your blog post as JSON object. Here is the format:
```
{output}
```
Each "Paragraph" text should be a brief overview or discussion (max 25 words) on a given topic from the list above. Your role is to create a colorful, engaging food blog post on pizza in this structured, straightforward format. Ready to cook up some delicious content?
"""

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.INFO
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("blogger")
    data = task.task()
    ask = data["blog"]
    llm = OpenAI()
    blog_post_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{blog_request}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=blog_post_prompt)
    ret = chain.invoke(
        {
            "blog_request": json.dumps(dict(blog=ask)),
            "input": "{'blog': ['Introduction: A few words about the history of pizza', 'Essential ingredients for pizza', 'Making the dough and sauce', 'Baking the pizza']}",
            "output": '{"answer":["Paragraph 1","Paragraph 2","Paragraph 3","Paragraph 4"]}',
        }
    )
    resp = json.loads(ret["text"].replace("\n", "").replace("System:", "").strip())
    assert task.answer(resp["answer"])
