import asyncio
import os

from dotenv import load_dotenv, find_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI()
semaphore = asyncio.Semaphore(5)  # Limit concurrency to 5


async def generate_description(doc):
    async with semaphore:
        system_message = SystemMessage(
            content="""
            Describe the following document with one of the following keywords:
            Mateusz, Jakub, Adam. Return the keyword and nothing else.
        """
        )
        human_message = HumanMessage(content=f"Document: {doc.page_content}")
        return await model.agenerate([[system_message], [human_message]])


# description_promises = [generate_description(doc) for doc in documents]
# descriptions = await asyncio.gather(*description_promises)


async def get_embeds(query):
    async with semaphore:
        embed = OpenAIEmbeddings()
        return await embed.aembed_query(query)


async def main():
    embed_promises = [get_embeds(q) for q in ["Mateusz", "Jakub", "Adam"]]
    embeds = await asyncio.gather(*embed_promises)
    print(len(embeds))


asyncio.run(main())
