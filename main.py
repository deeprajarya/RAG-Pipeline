from loader import load_and_chunk
from embedder import build_vector_store, search
from generator import generate_answer

def main():
    print("Initializing the RAG Pipeline.....")

    data_path = "data/Introduction to Graph Neural Networks.pdf"

    print("Loading and chunking the data")
    chunks = load_and_chunk(data_path)
    print(f"created {len(chunks)} chunks")

    print("Building vector store")
    client, model = build_vector_store(chunks)
    print("vector store is ready")

    print("\nRAG Pipeline is ready. Type quit to exit.")

    
    while True:
        query = input("Enter your question : ").strip()

        if query.lower() == "quit":
            print("exiting pipeline")
            break

        if query.lower() != "quit":
            

            print("\nsearching for relevant context")
            results = search(query, client, model)

            print("\nGenerating answer")
            answer = generate_answer(query, results)

            print(f"Answer :: \n{answer}")
            print("\n" + "─"*60 + "\n")

            continue

if __name__ == "__main__":
    main()