from core.core_functions import retrieve, synthesize, validate

def orchestrate(query):
    """
    Orchestrator = control layer of the system.

    Responsibilities:
    - Decide the execution flow of the pipeline
    - Coordinate core functions (retrieve → synthesize → validate)
    - Handle failure and retries
    - Ensure the system produces a reliable answer

    Flow:
    1. Start with conservative retrieval (k=3)
    2. Retrieve relevant chunks from vector DB
    3. Generate answer using retrieved context
    4. Validate answer quality
    5. If validation fails:
       - Increase retrieval depth (k=6)
       - Retry retrieval + synthesis
    6. Return final answer + supporting chunks
    """

    # Initial retrieval depth (balanced cost vs relevance)
    k = 3

    # Step 1 — retrieve relevant context
    chunks = retrieve(query, k)

    # Step 2 — generate answer from context
    answer = synthesize(query, chunks)

    # Step 3 — validate answer quality
    if not validate(answer):
        print("Retrying with k=6...")

        # Increase recall to improve chances of finding missing info
        k = 6

        # Retry retrieval with broader context
        chunks = retrieve(query, k)

        # Regenerate answer with new context
        answer = synthesize(query, chunks)

    # Final output returned to interface layer (main / API)
    return answer, chunks