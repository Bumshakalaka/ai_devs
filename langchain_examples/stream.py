import os

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")


chat = ChatOpenAI(streaming=True, callbacks=[MyCustomHandler()])

chat.invoke(
    [
        HumanMessage("Hey there!"),
    ]
)
