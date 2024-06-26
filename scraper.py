import logging
import os
import time

from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import OpenAI

from libs.TaskApi import TaskApi


context = """Foods similar to pizza have been made since the Neolithic Age. Records of people adding other ingredients to bread to make it more flavorful can be found throughout ancient history. In the 6th century BC, the Persian soldiers of the Achaemenid Empire during the rule of Darius the Great baked flatbreads with cheese and dates on top of their battle shields and the ancient Greeks supplemented their bread with oils, herbs, and cheese. An early reference to a pizza-like food occurs in the Aeneid, when Celaeno, queen of the Harpies, foretells that the Trojans would not find peace until they are forced by hunger to eat their tables (Book III). In Book VII, Aeneas and his men are served a meal that includes round cakes (like pita bread) topped with cooked vegetables. When they eat the bread, they realize that these are the "tables" prophesied by Celaeno.
The first mention of the word "pizza" comes from a notarial document written in Latin and dating to May 997 AD from Gaeta, demanding a payment of "twelve pizzas, a pork shoulder, and a pork kidney on Christmas Day, and 12 pizzas and a couple of chickens on Easter Day."Modern pizza evolved from similar flatbread dishes in Naples, Italy, in the 18th or early 19th century. Before that time, flatbread was often topped with ingredients such as garlic, salt, lard, and cheese. It is uncertain when tomatoes were first added and there are many conflicting claims. Until about 1830, pizza was sold from open-air stands and out of pizza bakeries.
A popular contemporary legend holds that the archetypal pizza, pizza Margherita, was invented in 1889, when the Royal Palace of Capodimonte commissioned the Neapolitan pizzaiolo (pizza maker) Raffaele Esposito to create a pizza in honor of the visiting Queen Margherita. Of the three different pizzas he created, the Queen strongly preferred a pizza swathed in the colors of the Italian flag — red (tomato), green (basil), and white (mozzarella). Supposedly, this kind of pizza was then named after the Queen, although later research cast doubt on this legend. An official letter of recognition from the Queen's "head of service" remains on display in Esposito's shop, now called the Pizzeria Brandi.Pizza was taken to the United States by Italian immigrants in the late nineteenth century and first appeared in areas where they concentrated. The country's first pizzeria, Lombardi's, opened in New York City in 1905. Following World War II, veterans returning from the Italian Campaign, who were introduced to Italy's native cuisine, proved a ready market for pizza in particular."""

system_prompt = """
Answer consistent (max 200 chars) in Polish on USER question using the context and nothing more.

context ```{context}```

"""

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    logger = logging.getLogger(__name__)
    loggerFormat = "%(asctime)s:\t%(message)s"
    loggerFormatter = logging.Formatter(loggerFormat)
    loggerLevel = logging.DEBUG
    logging.basicConfig(format=loggerFormat, level=loggerLevel)

    task = TaskApi("scraper")
    data = task.task()
    time.sleep(4.0)
    llm = OpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    ret = chain.invoke({"question": data["question"], "context": context})
    print(ret["text"])
    resp = ret["text"].replace("\n", "").replace("System:", "").strip()
    assert task.answer(resp)
