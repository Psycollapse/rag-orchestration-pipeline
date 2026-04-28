from openai import OpenAI
from core.core_functions import estimate_cost

client = OpenAI()


def direct_llm_answer(query):
    """
    Single-call baseline / fallback path.

    Purpose:
    - Answer when RAG retrieval is weak
    - Provide cost + quality comparison against RAG
    - Keep fallback responses concise and consistent with RAG answers
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question concisely in 3-5 sentences. "
                    "Match the style of short technical answers. "
                    "If multiple questions are present, answer them clearly and separately."
                ),
            },
            {"role": "user", "content": query},
        ],
    )

    answer = response.choices[0].message.content

    usage = response.usage
    cost = estimate_cost(usage)

    print("\n===== DIRECT LLM BASELINE =====")
    print(f"Cost: ${cost:.6f}")

    print("\n---Usage:---\n")
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

    print("\nAnswer:\n")
    print(answer)

    return answer, cost