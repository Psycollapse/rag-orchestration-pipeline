from core.core_functions import (
    retrieve,
    synthesize,
    validate,
    rewrite_query,
)
from control.query_checks import score_query


def orchestrate(query):
    """
    Orchestrator = control layer of the system.

    Responsibilities:
    - Handle multi-question queries
    - Evaluate query quality
    - Improve query when needed (rewrite logic)
    - Execute retrieval → synthesis pipeline
    - Evaluate retrieval quality
    - Handle validation failures with retry logic
    """

    # --- Multi-question handling ---
    if query.count("?") > 1:
        print("Multi-question detected → splitting...")

        questions = [q.strip() for q in query.split("?") if q.strip()]

        answers = []
        all_chunks = []

        for q in questions:
            sub_query = q + "?"
            print(f"\n--- Sub-question: {sub_query} ---")

            sub_query = rewrite_query(sub_query)

            chunks, scores = retrieve(sub_query, k=3)
            answer = synthesize(sub_query, chunks)

            answers.append(answer)
            all_chunks.extend(chunks)

        final_answer = "\n\n".join(answers)
        return final_answer, all_chunks

    # --- Initial scoring ---
    score = score_query(query)
    print(f"Query quality score: {score:.2f}")

    k = 3  # base retrieval depth

    # --- Query improvement layer ---
    if score < 0.4:
        print("Low quality query → rewriting...")
        query = rewrite_query(query)

    elif score < 0.7:
        print("Medium quality query → optional rewrite...")
        improved_query = rewrite_query(query)

        score_improved = score_query(improved_query)
        print(f"Improved query score: {score_improved:.2f}")

        if score_improved > score + 0.1:
            print(f"Using rewritten query: {improved_query}")
            query = improved_query
        else:
            print("Rewrite not significantly better → keeping original query")

    # --- Retrieval ---
    chunks, scores = retrieve(query, k)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Avg retrieval score: {avg_score:.2f}")

    # --- Retrieval quality check ---
    if avg_score < 0.5:
        print("Weak retrieval → rewriting query and retrying...")

        improved_query = rewrite_query(query)
        chunks, scores = retrieve(improved_query, k=5)

        avg_score_new = sum(scores) / len(scores) if scores else 0
        print(f"New avg retrieval score: {avg_score_new:.2f}")

        if avg_score_new > avg_score:
            print(f"Using improved query: {improved_query}")
            query = improved_query
        else:
            print("Rewrite did not improve retrieval → keeping original results")

    # --- Synthesis ---
    answer = synthesize(query, chunks)

    # --- Validation ---
    if not validate(answer):
        print("Validation failed → retrying with rewrite...")

        query = rewrite_query(query)
        chunks, _ = retrieve(query, k=6)
        answer = synthesize(query, chunks)

    return answer, chunks