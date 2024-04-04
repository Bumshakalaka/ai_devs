import logging
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

from libs.TaskApi import TaskApi

return_who_prompt = """
Napisz o osobę o którą chodzi w podanym zdaniu.
Zwróć mi tylko imię tej osoby i nic więcej.
Nie opdowiadaj na pytania, nie oczekuję innej odpowiedzi niż imię osoby

Przykład:
Zdanie: kim z zawodu jest Ernest?
Ernest

Zdanie: Ezaw ma kolor niebieskich oczu.
Ezaw

Zdanie: jakiego koloru oczy ma Ezaw?
Ezaw

Przeanalizuj zdanie

"""
answer_prompt = """
Odpowiedz na pytanie wykorzystując wiedzę zawartą tylko w podanym context.
Nie poprzedzaj odpowiedzi żadnym przedrostkiem, tylko zwróć odpowiedź i nic więcej.

context```{context}```
"""
if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.INFO
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("inprompt")
    data = task.task()
    inputs = data["input"]
    question = data["question"]

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", max_tokens=256, temperature=1)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", return_who_prompt),
            ("human", "{question}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    ret = chain.invoke(
        {
            "question": question,
        }
    )
    who = ret["text"].replace("\n", "").strip()
    context = []
    for a in inputs:
        if who.lower() in a.lower():
            context.append(a)
    print(ret)
    print(who)
    print(inputs)
    print(context)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_prompt),
            ("human", "{question}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    ret = chain.invoke({"question": question, "context": "\n".join(context)})
    print(ret["text"])
    task.answer(ret["text"].strip())
