import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# --- Init ---
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-pipeline")

DATA_DIR = "data"
total_cost = 0


# --- Load Documents ---
def load_docs():
    texts = []
    for file in os.listdir(DATA_DIR):
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


# --- Chunking (sentence-aware) ---
def chunk_text(text, size=800, overlap=200):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= size:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


# --- Embedding ---
def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# --- Indexing (run once) ---
def index_chunks():
    docs = load_docs()
    vectors = []
    id_counter = 0

    for doc in docs:
        chunks = chunk_text(doc)

        for chunk in chunks:
            embedding = embed_text(chunk)

            vectors.append({
                "id": str(id_counter),
                "values": embedding,
                "metadata": {"text": chunk}
            })

            id_counter += 1

    index.upsert(vectors=vectors)
    print(f"Indexed {len(vectors)} chunks.")


# --- Retrieval (UPDATED) ---
def retrieve(query, k):
    """
    Returns:
    - chunks: list[str]
    - scores: list[float]
    """

    query_embedding = embed_text(query)

    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )

    chunks = []
    scores = []

    for i, match in enumerate(results.matches):
        text = match.metadata.get("text", "")
        chunks.append(text)
        scores.append(match.score)

        print(f"[{i+1}] Score: {match.score:.3f}")

    return chunks, scores


# --- Synthesis ---
def synthesize(query, chunks):
    context = "\n\n".join(chunks)

    prompt = f"""
Answer the question using the context below.
Be concise and direct. Avoid unnecessary repetition.

Structure:
- Start with a clear, direct answer
- Add a brief explanation
- If needed, include a short note if something is inferred

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    usage = response.usage
    cost = estimate_cost(usage)

    global total_cost
    total_cost += cost

    print(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")

    with open("logs/usage.log", "a") as f:
        f.write(f"{usage.total_tokens},{cost}\n")

    print("\n---Usage:---\n")
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

    return response.choices[0].message.content


# --- Cost Estimation ---
def estimate_cost(usage):
    input_cost = usage.prompt_tokens * 0.15 / 1_000_000
    output_cost = usage.completion_tokens * 0.60 / 1_000_000
    return input_cost + output_cost


# --- Validation ---
def validate(answer):
    if len(answer) < 50:
        return False
    if "not enough" in answer.lower():
        return False
    return True


# --- Query Rewrite ---
def rewrite_query(query):
    prompt = f"""
Rewrite the following query to improve retrieval in a technical document search system.

Requirements:
- Preserve ALL original concepts
- Convert keywords into a clear natural-language question
- Be explicit and specific

Original:
{query}

Rewritten:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    rewritten = response.choices[0].message.content.strip()
    print(f"Rewritten query: {rewritten}")

    return rewritten