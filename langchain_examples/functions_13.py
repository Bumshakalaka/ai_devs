import logging
import os
import pprint

from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def get_llm_kwargs(function: dict) -> dict:
    """Returns the kwargs for the LLMChain constructor.

    Args:
        function: The function to use.

    Returns:
        The kwargs for the LLMChain constructor.
    """
    return {"functions": [function], "function_call": {"name": function["name"]}}


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    function = {
        "name": "query_enrichment",
        "description": "Describe users query with semantic tags and classify with type",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "boolean",
                    "description": "Set to 'true' when query is direct command for AI. Set to 'false' when queries asks for saying/writing/translating/explaining something and all other.",
                },
                "type": {
                    "type": "string",
                    "description": "memory (queries about the user and/or AI), notes|links (queries about user's notes|links). By default pick 'memory'.",
                    "enum": ["memory", "notes", "links"],
                },
                "tags": {
                    "type": "array",
                    "description": "Multiple semantic tags/keywords that enriches query for search purposes (similar words, meanings). When query refers to the user, add 'overment' tag, and when refers to 'you' add tag 'Alice'",
                    "items": {"type": "string"},
                },
            },
            "required": ["type", "tags", "command"],
        },
    }

    system_template = """
    Describe users query with semantic tags and classify with type.
    command: Set to 'true' when query is direct command for AI. Set to 'false' when queries asks for saying/writing/translating/explaining something and all other.
    type: memory (queries about the user and/or AI), notes|links (queries about user's notes|links). By default pick 'memory'.
    tags: Multiple semantic tags/keywords that enriches query for search purposes (similar words, meanings). When query refers to the user, add 'overment' tag, and when refers to 'you' add tag 'Alice'
    """

    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", human_template),
        ]
    )

    output_parser = JsonOutputFunctionsParser()
    llm_kwargs = get_llm_kwargs(function)
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        tags=None,
        verbose=True,
    )

    ret = chain.invoke(
        input=dict(
            text="Save information that LanChain is very good Python package for LLMs."
        )
    )
    pprint.pprint(ret)

    ret = chain.invoke(input=dict(text="Hi there"))
    pprint.pprint(ret)
