import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import numpy as np

# --- Init ---
# Load environment variables (API keys, etc.)
load_dotenv()

# Initialize OpenAI client for embeddings + LLM
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone vector DB
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-pipeline")

# Local dataset directory
DATA_DIR = "data"

# Running cost tracker (for observability)
total_cost = 0


# --- Load Documents ---
def load_docs():
    """
    Reads all files from /data and returns list of raw text documents.
    """
    texts = []
    for file in os.listdir(DATA_DIR):
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


# --- Chunking (sentence-aware + overlap) ---
def chunk_text(text, size=800, overlap=200):
    """
    Splits text into chunks while preserving sentence boundaries.

    Why:
    - Smaller chunks → better precision
    - Overlap → avoids losing context at boundaries
    """

    # Split into sentences using regex
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for sentence in sentences:
        # If adding sentence stays within size → keep accumulating
        if len(current) + len(sentence) <= size:
            current += " " + sentence
        else:
            # Save chunk
            chunks.append(current.strip())

            # Apply overlap (carry last N chars forward)
            current = current[-overlap:] + " " + sentence

    if current:
        chunks.append(current.strip())

    return chunks


# --- Embedding ---
def embed_text(text):
    """
    Converts text into embedding vector using OpenAI.

    This is used for:
    - indexing chunks
    - embedding queries
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# --- Indexing (RUN ONCE) ---
def index_chunks():
    """
    Loads docs → chunks → embeds → stores in Pinecone.

    IMPORTANT:
    We now store embeddings ALSO in metadata so we can reuse them later
    (avoids recomputing during reranking).
    """
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
                "metadata": {
                    "text": chunk
                }
            })

            id_counter += 1

    index.upsert(vectors=vectors)
    print(f"Indexed {len(vectors)} chunks.")


# --- Retrieval ---
def retrieve(query, k):
    """
    Retrieves top-k most similar chunks from Pinecone.

    Returns:
    - chunks (text)
    - scores (pinecone similarity)
    - embeddings (cached embeddings from metadata)
    """

    # Embed query
    query_embedding = embed_text(query)

    # Query vector DB
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )

    if not results.matches:
        print("No matches found.")
        return [], [], []

    chunks = []
    scores = []
    embeddings = []

    for i, match in enumerate(results.matches):
        text = match.metadata.get("text", "")
        emb = match.metadata.get("embedding")

        chunks.append(text)
        scores.append(match.score)
        embeddings.append(emb)

        print(f"[{i+1}] Score: {match.score:.3f}")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Avg Score: {avg_score:.3f}")

    return chunks, scores, query_embedding


# --- Synthesis ---
def synthesize(query, chunks):
    """
    Generates answer using retrieved chunks as context.

    The LLM is instructed to:
    - prefer context
    - be concise
    - avoid hallucination
    """

    context = "\n\n".join(chunks) if chunks else "No relevant context retrieved."

    prompt = f"""
Answer the question using the context below.

Rules:
- Prefer using the context
- Be concise
- If weak context, answer conservatively

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

    return response.choices[0].message.content


# --- Cost Estimation ---
def estimate_cost(usage):
    """
    Estimates cost based on token usage.
    """
    input_cost = usage.prompt_tokens * 0.15 / 1_000_000
    output_cost = usage.completion_tokens * 0.60 / 1_000_000
    return input_cost + output_cost


# --- Validation ---
def validate(answer):
    """
    Simple heuristic validation:
    - Too short → bad
    - "not enough" → bad
    """
    if len(answer) < 50:
        return False
    if "not enough" in answer.lower():
        return False
    return True


# --- Query Rewrite ---
def rewrite_query(query):
    """
    Improves query clarity for better retrieval.
    """
    prompt = f"""
Rewrite the query to improve retrieval.

Original:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    rewritten = response.choices[0].message.content.strip()
    print(f"Rewritten query: {rewritten}")
    return rewritten


# --- Cosine Similarity ---
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Used for semantic reranking.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))