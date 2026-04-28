from core.core_functions import (
    retrieve,
    synthesize,
    validate,
    rewrite_query,
    embed_text,
    cosine_similarity,
)
from control.query_checks import score_query
import numpy as np


# =========================
# NORMALIZATION HELPER
# =========================
def normalize_scores(scores):
    """
    Normalize scores into [0, 1] using min-max scaling.

    Why:
    - Pinecone scores and cosine similarity are on different scales
    - We need them comparable before combining
    """
    if not scores:
        return scores

    min_s = min(scores)
    max_s = max(scores)

    # Avoid division by zero (all scores equal)
    if max_s == min_s:
        return [0.5 for _ in scores]

    return [(s - min_s) / (max_s - min_s) for s in scores]


# =========================
# MAIN ORCHESTRATOR
# =========================
def orchestrate(query):
    """
    Orchestrator = control layer of the system.

    Responsibilities:
    - Detect multi-question queries
    - Perform shared retrieval
    - Apply semantic reranking (recomputing embeddings)
    - Apply hybrid scoring (semantic + Pinecone)
    - Apply dynamic confidence threshold
    - Route between RAG and fallback (LLM)
    - Validate answers post-generation
    - Return structured output
    """

    # =========================
    # MULTI-QUESTION HANDLING
    # =========================
    if query.count("?") > 1:
        print("Multi-question detected → splitting...")

        # Split query into sub-questions
        questions = [q.strip() for q in query.split("?") if q.strip()]

        # -------------------------
        # Step 1: Combined retrieval
        # -------------------------
        combined_query = rewrite_query(" ".join(questions))

        # -------------------------
        # Step 2: Shared retrieval
        # -------------------------
        # Pinecone returns:
        # - chunks (text)
        # - scores (similarity)
        # - query_embedding (we reuse this)
        chunks, pinecone_scores, query_embedding = retrieve(combined_query, k=6)

        answers = []

        # =========================
        # Step 3: Per-question processing
        # =========================
        for q in questions:
            sub_query = q + "?"
            print(f"\n--- Sub-question: {sub_query} ---")

            # -------------------------
            # Embed sub-query
            # -------------------------
            sub_query_embedding = embed_text(sub_query)

            # -------------------------
            # Semantic similarity (recomputed)
            # -------------------------
            semantic_scores = []

            for chunk in chunks:
                # IMPORTANT:
                # Pinecone does NOT return stored embeddings
                # so we recompute them here
                chunk_embedding = embed_text(chunk)

                sim = cosine_similarity(sub_query_embedding, chunk_embedding)
                semantic_scores.append(sim)

            # -------------------------
            # Normalize scores
            # -------------------------
            norm_semantic = normalize_scores(semantic_scores)
            norm_pinecone = normalize_scores(pinecone_scores)

            # -------------------------
            # Hybrid scoring
            # -------------------------
            alpha = 0.7  # semantic weight

            scored_chunks = []

            for chunk, sem_s, pine_s in zip(chunks, norm_semantic, norm_pinecone):
                hybrid_score = alpha * sem_s + (1 - alpha) * pine_s
                scored_chunks.append((chunk, hybrid_score))

            # -------------------------
            # Sort chunks by relevance
            # -------------------------
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            # Select top-k chunks for this question
            top_chunks = scored_chunks[:3]
            filtered_chunks = [c for c, _ in top_chunks]

            # =========================
            # DYNAMIC THRESHOLD
            # =========================
            all_scores = [s for _, s in scored_chunks]

            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)

            beta = 0.3  # stricter threshold

            dynamic_threshold = mean_score - beta * std_score

            avg_local_score = sum(s for _, s in top_chunks) / len(top_chunks)

            print(
                f"Mean: {mean_score:.3f} | Std: {std_score:.3f} | "
                f"Threshold: {dynamic_threshold:.3f} | Top-k avg: {avg_local_score:.3f}"
            )

            # =========================
            # ROUTING DECISION
            # =========================
            if avg_local_score < dynamic_threshold or len(filtered_chunks) < 2:
                print("Dynamic threshold triggered → fallback to direct LLM")

                from control.evaluator import direct_llm_answer
                answer, _ = direct_llm_answer(sub_query)

            else:
                # -------------------------
                # RAG synthesis
                # -------------------------
                answer = synthesize(sub_query, filtered_chunks)

                # -------------------------
                # Answer-aware validation
                # -------------------------
                if (
                    not validate(answer)
                    or "not found" in answer.lower()
                    or "does not" in answer.lower()
                    or "can be inferred" in answer.lower()
                ):
                    print("Weak answer → fallback to direct LLM")

                    from control.evaluator import direct_llm_answer
                    answer, _ = direct_llm_answer(sub_query)

            answers.append(answer)

        # =========================
        # FORMAT FINAL OUTPUT
        # =========================
        final_answer = ""

        for i, (q, a) in enumerate(zip(questions, answers), 1):
            final_answer += f"{i}. {q}?\n{a.strip()}\n\n"

        return final_answer, chunks

    # =========================
    # SINGLE QUERY FLOW
    # =========================

    score = score_query(query)
    print(f"Query quality score: {score:.2f}")

    k = 3

    # -------------------------
    # Query improvement
    # -------------------------
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

    # -------------------------
    # Retrieval
    # -------------------------
    chunks, scores, _ = retrieve(query, k)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Avg retrieval score: {avg_score:.2f}")

    # -------------------------
    # Routing
    # -------------------------
    if avg_score < 0.45:
        print("Low retrieval confidence → fallback to direct LLM")

        from control.evaluator import direct_llm_answer
        return direct_llm_answer(query)

    # -------------------------
    # Synthesis
    # -------------------------
    answer = synthesize(query, chunks)

    # -------------------------
    # Validation
    # -------------------------
    if not validate(answer):
        print("Validation failed → fallback to direct LLM")

        from control.evaluator import direct_llm_answer
        return direct_llm_answer(query)

    return answer, chunks