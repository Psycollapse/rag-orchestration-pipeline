import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv() #load environment variables from .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-pipeline") #the index in pinecone its literally the database with the vectors that relate how similar each chunk of text is to other chunks of text

DATA_DIR = "data" #the directory where the documents are stored
total_cost = 0 #we'll want to track the total cost of all API calls

def load_docs():
    '''
    - list files in /data
    - open each file
    - read content
    - append to list
    - return list
    '''
    texts = []
    for file in os.listdir(DATA_DIR): #we will iterate through all files in the data directory
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f: #we open each file in read mode with utf-8 encoding
            texts.append(f.read()) #we append the content of each file to the texts list
    return texts #we need to return content because we want to use it in other functions.
    # since we finished the function, we can comment the "pass" to avoid confusion (and delete it entirely in next functions)
    #pass

def chunk_text(text, size=800, overlap =200):
    """
    initialize index i = 0
    slice text[i:i+size]
    append chunk
    increment i += size - overlap
    repeat until end
    """
    chunks = [] #we initialize an empty list to store the chunks
    i=0 #we initialize the index i to 0
    while i < len(text): #we iterate through the text until we reach the end
        chunk = text[i:i+size] #we slice the text from index i to i+size
        chunks.append(chunk) #we append the chunk to the chunks list
        i += size - overlap #we increment i by size - overlap to get the next chunk
    return chunks #we return the chunks list


def embed_text(text):
    #TODO
    """
    used for chunks and for query
    embed text
    return embedding
    """
    response = client.embeddings.create(input=text, model="text-embedding-3-small") #embedding means converting text to a vector representation of how similar it is to other texts
    return response.data[0].embedding #we return the index 0 of the data array which contains the embedding, a list of numbers that relate how much each word in the text is similar to other texts



def index_chunks():
    #TODO
    """
    Run ONCE or when data changes (docs)
    embed chunks
    store in vector db
    return vector db
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
                "metadata": {"text": chunk}
        
            })
            id_counter += 1
    
    index.upsert(vectors=vectors)
    print(f"Indexed {len(vectors)} chunks.")


def retrieve(query, k):
    """
    PAST VERSION OF RETRIEVE 
    load docs
    chunk each doc
    combine all chunks
    return first k chunks
    docs = load_docs() #we load the documents
    all_chunks = [] #we initialize an empty list to store all chunks
    
    for doc in docs: #we iterate through all documents
        all_chunks.extend(chunk_text(doc)) #if we used append instead of extend, our problem would be that we would be adding a list to the list instead of individual chunks
    return all_chunks[:k] #we return the first k chunks
    """

    """
    NEWER VERSION OF RETRIEVE
    embed_query = embed_text(query)
    matches = pinecone.search(embed_query, k=k)
    return matches
    """

    query_embedding = embed_text(query)

    results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
    chunks = [match.metadata['text'] for match in results.matches]
    return chunks

def synthesize(query, chunks):
    """
    join chunks into context string
    build prompt (context + query)
    call LLM
    return answer
    """
    context = "\n\n".join(chunks) #we join the chunks with a double newline
    
    prompt = f"""
    Answer the question using the context below.
    Be concise and direct. Avoid unnecessary repetition.

    Structure:
    - Start with a clear, direct answer
    - Add a brief explanation
    - If needed, include a short note indicating whether any part is inferred beyond the

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create( #we call the OpenAI API to get a response
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    usage = response.usage #before returning, we will also get the usage and cost
    cost = estimate_cost(usage)
    
    global total_cost
    total_cost += cost
    
    print(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    
    with open("logs/usage.log","a") as f:
        f.write(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}\n")
    
    print("\n---Usage:---\n")
    print(f"Input/Prompt tokens: {usage.prompt_tokens}") #tokens are a measure of text length that the model has to process
    print(f"Output/Completion tokens: {usage.completion_tokens}") #tokens are a measure of text length that the model generates
    print(f"Total tokens: {usage.total_tokens}") #total tokens is the sum of input and output tokens

    return response.choices[0].message.content #the [0] is to get the first choice from the list of choices of the response

def estimate_cost(usage): #we add this to estimate the cost of the API calls
    input_cost = usage.prompt_tokens * 0.15 / 1_000_000 #we calculate the input cost
    output_cost = usage.completion_tokens * 0.60 / 1_000_000 #we calculate the output cost
    return input_cost + output_cost #we return the total cost

def validate(answer):
    """
    check if too short
    check if “not enough info”
    return True / False
    """
    if len(answer) < 50: #if the answer is too short, it's not valid
        return False
    if "not enough" in answer.lower(): #if the answer contains "not enough", it's not valid
        return False
    return True #otherwise, it's valid