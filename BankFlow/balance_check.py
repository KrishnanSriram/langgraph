from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph


class AccountBalanceState(TypedDict):
    account_number: str
    balance: float
    response: str


def check_balance_operations(data: AccountBalanceState):
    print(f"Simulating checking balance for account: {data.get('account_number')}")
    # In a real system, you would interact with a bank API here.
    # For this example, we'll return a static balance based on the account.
    if data.get("account_number") == "ACC123":
        return {"balance": 1500.75}
    elif data.get("account_number") == "ACC456":
        return {"balance": 500.20}
    else:
        return {"balance": -1.0, "response": "Error: Account not found."}



def generate_balance_response(data: AccountBalanceState):
    model = ChatOllama(model="gemma3:12b") # Replace with your preferred Ollama model

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful banking assistant."),
            ("human", "The balance for account {account_number} is: {balance}"),
        ]
    )
    chain = prompt | model | StrOutputParser()
    response = chain.invoke(data)
    return {"response": response}


def log_balance_check(data: AccountBalanceState):
    print("--- Balance Check Activity ---")
    print(f"Balance for account {data.get('account_number')}: ${data.get('balance', 'N/A'):.2f}")
    print(f"Response to user: {data.get('response')}")
    return data


def create_check_balance_workflow() -> CompiledStateGraph:
    check_balance_workflow = StateGraph(AccountBalanceState)
    check_balance_workflow.add_node("check_balance", check_balance_operations)
    check_balance_workflow.add_node("generate_response", generate_balance_response)
    check_balance_workflow.add_node("log_activity", log_balance_check)

    # Define the edges for checking balance
    check_balance_workflow.set_entry_point("check_balance")
    check_balance_workflow.add_edge("check_balance", "generate_response")
    check_balance_workflow.add_edge("generate_response", "log_activity")
    check_balance_workflow.add_edge("log_activity", END)

    return check_balance_workflow.compile()


def main():
    # Define the graph for checking balance
    check_balance_chain = create_check_balance_workflow()

    check_balance_request = {"user_id": "user123", "account_number": "ACC123"}
    # invalid_account_request = {"user_id": "user456", "account_number": "ACC789"}

    print("\n--- Checking Balance ---")
    balance_result = check_balance_chain.invoke(check_balance_request)
    print(balance_result)


if __name__ == "__main__":
    main()