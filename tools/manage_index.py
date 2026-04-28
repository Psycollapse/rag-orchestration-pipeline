from pinecone import Pinecone, ServerlessSpec
from config.settings import INDEX_NAME, DIMENSION, METRIC
from core.core_functions import index_chunks

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from core.core_functions import index_chunks

# --- Init ---
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "rag-pipeline"
DIMENSION = 1536  # text-embedding-3-small
METRIC = "cosine"


def index_exists():
    indexes = pc.list_indexes()
    return any(i["name"] == INDEX_NAME for i in indexes)


def create_index():
    print(f"Creating index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


def delete_all_vectors():
    print("Deleting all vectors from index...")
    index = pc.Index(INDEX_NAME)
    index.delete(delete_all=True)
    print("Index cleared.")


def main():
    print("\n=== Pinecone Index Manager ===\n")

    # --- Check existence ---
    if not index_exists():
        print(f"Index '{INDEX_NAME}' does NOT exist.")

        create = input("Create it? (y/n): ").lower()

        if create == "y":
            create_index()
        else:
            print("Aborting.")
            return

    else:
        print(f"Index '{INDEX_NAME}' already exists.")

        wipe = input("Delete existing data before indexing? (y/n): ").lower()

        if wipe == "y":
            delete_all_vectors()

    # --- Run indexing ---
    print("\nStarting indexing...\n")
    index_chunks()

    print("\nDone.\n")


if __name__ == "__main__":
    main()