# main.py
# Entry point of the application.
# This file is responsible only for interacting with the user (CLI for now).
# All processing logic is delegated to the orchestrator.

from control.orchestrator import orchestrate

def main():
    """
    Flow:
    1. Receive user query
    2. Delegate to orchestrator (control layer)
    3. Orchestrator handles:
       - retrieval
       - synthesis
       - validation
       - retries if needed
    4. Print final answer and supporting chunks
    """

    query = input("Ask: ")  # user input

    answer, chunks = orchestrate(query)  # delegate full pipeline

    print("\nAnswer:\n")
    print(answer)

    print("\nRetrieved chunks:\n")
    print(chunks)


if __name__ == "__main__":
    main()