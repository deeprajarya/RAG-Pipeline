from ollama import chat

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([chunk.payload["text"] for chunk in retrieved_chunks])
    prompt = f"""You are a helpful assistant. Use only the context 
below to answer the question. If the answer is not found in the 
context, say "I don't know based on the provided documents.

context: {context}
question: {query}
answer:

"""
    response = chat(
        model="mistral", 
        messages=[{"role": "user", "content": prompt}])
    return response.message.content

if __name__ == "__main__":
    from embedder import build_vector_store, search
    from loader import load_and_chunk

    chunks = load_and_chunk("data/Introduction to Graph Neural Networks.pdf")
    client, model = build_vector_store(chunks)

    query = "What is a graph neural network?"
    results = search(query, client, model)
    answer = generate_answer(query, results)
    
    print(f"\nQuery: {query}")
    print(f"\nAnswer: {answer}")
