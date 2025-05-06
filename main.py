from itertools import chain
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
import datetime
import json

# 1. Define the state
class UserInfo(TypedDict):
    name: str
    gender: str
    location: str
    time_of_day: str
    greeting: str


def determine_time_of_day(data: UserInfo):
    now = datetime.datetime.now()
    hour = now.hour
    if 5 <= hour < 12:
        return {"time_of_day": "morning"}
    elif 12 <= hour < 17:
        return {"time_of_day": "afternoon"}
    elif 17 <= hour < 21:
        return {"time_of_day": "evening"}
    else:
        return {"time_of_day": "night"}

def generate_greeting(data: UserInfo):
    model = ChatOllama(model="gemma3:12b") # Replace with your preferred Ollama model

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly assistant that greets people based on the time of day and their gender.",
            ),
            (
                "human",
                "Greet {name}. It is currently {time_of_day} in {location}.",
            ),
        ]
    )
    chain = prompt | model | StrOutputParser()
    greeting = chain.invoke(data)
    return {"greeting": greeting}

def log_greeting(data: UserInfo):
    print(f"Greeting generated for {data['name']}: {data['greeting']}")
    return data


def define_workflow():
    workflow = StateGraph(UserInfo)
    workflow.add_node("determine_time", determine_time_of_day)
    workflow.add_node("generate_greeting", generate_greeting)
    workflow.add_node("log_greeting", log_greeting)

    # 4. Define the edges
    workflow.set_entry_point("determine_time")
    workflow.add_edge("determine_time", "generate_greeting")
    workflow.add_edge("generate_greeting", "log_greeting")
    workflow.add_edge("log_greeting", END)
    return workflow

def main():
    gchain = define_workflow().compile()
    user_data = {"name": "Alice", "gender": "female", "location": "Dublin, Ohio"}

    # 7. Run the chain
    result = gchain.invoke(user_data)
    print(result)

if __name__ == "__main__":
    main()