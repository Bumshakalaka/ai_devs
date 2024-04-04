import logging
import os
import pprint
from typing import List, Type, Any

from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


class AddSchema(BaseModel):
    """Add two numbers."""

    first: int = Field(
        ...,
        description="First value to add",
    )

    second: int = Field(
        ...,
        description="Second value to add",
    )


class Add(BaseTool):
    args_schema: Type[BaseModel] = AddSchema
    name: str = "add"
    description: str = "Add two numbers."

    def _run(self, a: int, b: int, **kwargs: Any) -> Any:
        return a + b


class MultiplySchema(BaseModel):
    """Multiply two numbers."""

    first: int = Field(
        ...,
        description="First value to Multiply",
    )

    second: int = Field(
        ...,
        description="Second value to Multiply",
    )


class Multiply(BaseTool):
    args_schema: Type[BaseModel] = MultiplySchema
    name: str = "multiply"
    description: str = "Multiply two numbers."

    def _run(self, a: int, b: int, **kwargs: Any) -> Any:
        return a * b


class SubtractSchema(BaseModel):
    """Subtract two numbers."""

    first: int = Field(
        ...,
        description="First value to subtract",
    )

    second: int = Field(
        ...,
        description="Second value to subtract",
    )


class Subtract(BaseTool):
    args_schema: Type[BaseModel] = SubtractSchema
    name: str = "subtract"
    description: str = "subtract two numbers."

    def _run(self, a: int, b: int, **kwargs: Any) -> Any:
        return a - b


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose=True)
    llm_with_tools = llm.bind_tools([AddSchema, SubtractSchema, MultiplySchema])
    tool_chain = llm_with_tools | PydanticToolsParser(
        tools=[AddSchema, SubtractSchema, MultiplySchema]
    )
    print(tool_chain.invoke("2+4"))
    print(tool_chain.invoke("2-4"))
    print(tool_chain.invoke("44*5.1"))
