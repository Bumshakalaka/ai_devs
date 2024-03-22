import asyncio
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

handler = StdOutCallbackHandler()
llm = OpenAI()
chain = LLMChain(
    llm=llm, prompt=PromptTemplate.from_template("Hi there"), callbacks=[handler]
)


# Definicja funkcji asynchronicznej do wykonania chatu
async def main():
    await chain.ainvoke({})


# Uruchomienie funkcji asynchronicznej w pętli zdarzeń
asyncio.run(main())
