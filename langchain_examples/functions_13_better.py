import logging
import os
import pprint
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class Query_enrichment(BaseModel):
    """Describe the user's query with semantic tags and classify with type."""

    command: bool = Field(
        ...,
        description="Set to 'true' when query is direct command for AI. Set to 'false' when queries asks for saying/writing/translating/explaining something and all other.",
    )
    type: str = Field(
        ...,
        description="memory (queries about the user and/or AI), notes|links (queries about user's notes|links). By default pick 'memory'.",
    )
    tags: List[str] = Field(
        ...,
        description="Multiple semantic tags/keywords that enriches query for search purposes (similar words, meanings). When query refers to the user, add 'user' tag, and when refers to 'you' add tag 'AI'",
    )


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose=True)
    llm_with_tools = llm.bind_tools([Query_enrichment])
    tool_chain = llm_with_tools | PydanticToolsParser(tools=[Query_enrichment])
    ret = tool_chain.invoke(
        "Save information that LanChain is very good Python package for LLMs."
    )
    pprint.pprint(ret)
