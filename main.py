# main.py
# Entry point of the application.
# Handles CLI only. All logic is delegated to the orchestrator.

from control.orchestrator import orchestrate


def main():
    """
    Flow:
    1. Receive user query
    2. Delegate to orchestrator
    3. Print final answer
    4. Optionally inspect retrieved chunks (debug)
    """

    query = input("Ask: ").strip()

    print("\n===== RAG PIPELINE =====")

    answer, chunks = orchestrate(query)

    # --- Final Answer ---
    print("\n===== FINAL ANSWER =====\n")
    print(answer)

    # --- Debug: Retrieved Chunks ---
    print("\n===== RETRIEVED CHUNKS (DEBUG) =====\n")

    for i, chunk in enumerate(chunks[:5]):  # limit output
        print(f"[{i+1}] {chunk[:200]}...\n")  # truncate for readability


if __name__ == "__main__":
    main()