import os

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicjalizacja domyślnego modelu, czyli gpt-3.5-turbo
chat = ChatOpenAI()
# Wywołanie modelu poprzez przesłanie tablicy wiadomości.
# W tym przypadku to proste przywitanie
messages = [HumanMessage("Hey there!")]
response = chat.invoke(messages)

print(response.content)
