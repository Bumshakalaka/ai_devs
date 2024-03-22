import asyncio
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAI

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat = OpenAI()
memory = ConversationBufferWindowMemory(k=1)
chain = ConversationChain(llm=chat, memory=memory)


async def main():
    response1 = await chain.ainvoke(input="Hey there! I'm Adam")
    print("AI:", response1)  # Hi Adam!

    response2 = await chain.ainvoke(input="Hold on.")
    print("AI:", response2)  # Likewise, how can I help you?

    # Here the model "forgets" the name because "k" is set to 1. The earlier message was truncated.
    response3 = await chain.ainvoke(input="Do you know my name?")
    print("AI: ", response3)  # Nope.


asyncio.run(main())
