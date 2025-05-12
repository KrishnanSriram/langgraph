from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph


class TransferFundsState(TypedDict):
    user_id: str
    account_from: str
    account_to: str
    transfer_amount: float
    transfer_result: str
    response: str


def initiate_transfer_operations(data: TransferFundsState):
    print(f"Simulating initiating transfer of ${data.get('transfer_amount')} from {data.get('account_from')} to {data.get('account_to')}")
    # In a real system, you would interact with a bank API to initiate the transfer.
    # For this example, we'll just simulate success.
    if data.get("transfer_amount", 0) > 0:
        return {"transfer_result": "Transfer initiated successfully."}
    else:
        return {"transfer_result": "Error: Invalid transfer amount."}


def generate_transfer_response(data: TransferFundsState):
    model = ChatOllama(model="gemma3:12b")  # Replace with your preferred Ollama model

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful banking assistant."),
            ("human", "{transfer_result}"),
        ]
    )
    chain = prompt | model | StrOutputParser()
    response = chain.invoke(data)
    return {"response": response}


def log_transfer_activity(data: TransferFundsState):
    print("--- Transfer Activity ---")
    print(f"Transfer from: {data.get('account_from')}")
    print(f"Transfer to: {data.get('account_to')}")
    print(f"Amount: ${data.get('transfer_amount', 'N/A'):.2f}")
    print(f"Result: {data.get('transfer_result')}")
    print(f"Response to user: {data.get('response')}")
    return data


def create_transfer_workflow() -> CompiledStateGraph:
    transfer_funds_workflow = StateGraph(TransferFundsState)
    transfer_funds_workflow.add_node("initiate_transfer", initiate_transfer_operations)
    transfer_funds_workflow.add_node("generate_response", generate_transfer_response)
    transfer_funds_workflow.add_node("log_activity", log_transfer_activity)

    # 4. Define the edges for transferring funds
    transfer_funds_workflow.set_entry_point("initiate_transfer")
    transfer_funds_workflow.add_edge("initiate_transfer", "generate_response")
    transfer_funds_workflow.add_edge("generate_response", "log_activity")
    transfer_funds_workflow.add_edge("log_activity", END)

    # 5. Compile the transfer funds graph
    return transfer_funds_workflow.compile()


def main():
    # Define the graph for checking balance
    transfer_funds_chain = create_transfer_workflow()

    transfer_request = {"user_id": "user123", "account_from": "ACC123", "account_to": "ACC456",
                        "transfer_amount": 100.00}
    invalid_transfer_request = {"user_id": "user789", "account_from": "ACC111", "account_to": "ACC222",
                                "transfer_amount": -50.00}
    missing_amount_request = {"user_id": "user123", "account_from": "ACC123", "account_to": "ACC456"}

    # 7. Run the transfer funds chain
    print("\n--- Initiating Transfer ---")
    transfer_result = transfer_funds_chain.invoke(transfer_request)
    print(transfer_result)

    print("\n--- Initiating Transfer - Invalid Amount ---")
    invalid_transfer_result = transfer_funds_chain.invoke(invalid_transfer_request)
    print(invalid_transfer_result)

    print("\n--- Initiating Transfer - Missing Amount ---")
    missing_amount_result = transfer_funds_chain.invoke(missing_amount_request)
    print(missing_amount_result)


if __name__ == "__main__":
    main()