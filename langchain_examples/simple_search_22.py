import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    documents = [
        Document(page_content="Adam is a programmer."),
        Document(page_content="Adam has a dog named Alexa."),
        Document(page_content="Adam is also a designer."),
    ]
    db = FAISS.from_documents(documents, OpenAIEmbeddings())

    query = "Who is Adam?"
    docs = db.similarity_search(query, k=1)
    print(docs)
