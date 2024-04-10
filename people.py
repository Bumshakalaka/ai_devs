import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient

from libs.TaskApi import TaskApi


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    COLLECTION_NAME = "unknownNews"

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.INFO
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("people")
    query = task.task()["question"]

    chat = ChatOpenAI(model="gpt-4", temperature=1.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Hej, jestem asystentem, który umie idealnie kategoryzować i wyciągać informacje ze zdania i nic wiecej.
                
                Moje zasady wyciągania informacji:
                - name: Imię osoby o która chodzi w pytaniu. Zamień na oficjalna formę w mianowniku, bez zdrobnień
                - surname: Nazwisko osoby o która chodzi w pytaniu. Zamień na mianownik
                - what: O co jest pytanie. Możliwe kategorie, to:
                  - color: pytanie o kolor
                  - food: pytanie o jedzenie
                  - address: pytanie o miejsce zamieszkania, o miasto lub adres
                
                Zawsze zwracam odpowiedź w formacie JSON: {json_example}
                
                Przykłady:
                {examples}             
                """,
            ),
            ("human", "{query}"),
        ]
    )
    examples = """
                User: powiedz mi, gdzie mieszka Katarzyna Truskawka? w jakim mieście?
                AI: {"name":"Katarzyna", "surname":"Truskawka", "what":"address"}
                
                User: jaki kolor się podoba Mariuszowi Kaczorowi?
                AI: {"name":"Mariusz", "surname":"Kaczor", "what":"color"}
                
                User: co lubi jeść Tomek Bzik?
                AI: {"name":"Tomasz", "surname":"Bzik", "what":"food"}"""
    json_example = '{"name":"Katarzyna", "surname":"Truskawka", "what":"address"}'
    chain = prompt | chat
    ret = chain.invoke(
        {
            "query": query,
            "examples": examples,
            "json_example": json_example,
        }
    )
    print(query)
    print(ret)
    result = json.loads(ret.content)
    with open(Path(__file__).parent / "people.json", "r") as fd:
        db = json.load(fd)
    context = next(
        x
        for x in db
        if x["imie"] == result["name"] and x["nazwisko"] == result["surname"]
    )
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Hej, jestem asystentem, który umie zwięźle i krótko odpowiadać na pytania o ulubiony kolor, jedzenie i miejsce zamieszkania i nic więcej.
                Odpowiadam pod uwagę tylko poniższy kontekst w formacie JSON
                Jeżeli nie znam odpowiedzi, zwracam `nie wiem`
                
                context```{context}```            
                """,
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | chat
    ret = chain.invoke(
        {
            "query": query,
            "context": context,
        }
    )
    print(ret)
    task.answer(ret.content)
