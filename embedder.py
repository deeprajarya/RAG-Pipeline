'''
Two jobs of embedder.py are:

1. Create embeddings for all chunks and store them in Qdrant
2. Create an embedding for a query and search Qdrant for similar chunks'''


from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def build_vector_store(chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="pdf_chunks",
        vectors_config=VectorParams(size = 384, distance=Distance.COSINE)
        )
    texts = []
    for chunk in chunks:
        texts.append(chunk.page_content)

    points = []
    embeddings = model.encode(texts)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=i,
            vector = embedding.tolist(),
            payload ={
                "text":chunk.page_content,
                "page":chunk.metadata["page"],
                "source":chunk.metadata["source"],
            }
            ))

    client.upsert(collection_name="pdf_chunks", points=points)
    return client, model

def search(query, client, model, top_k=5):
    query_embedding = model.encode(query).tolist()
    results = client.query_points(
        collection_name="pdf_chunks",
        query=query_embedding,
        limit=top_k,
    )
    return results.points




if __name__ == "__main__":
    from loader import load_and_chunk

    chunks = load_and_chunk("data/Introduction to Graph Neural Networks.pdf")
    client, model = build_vector_store(chunks)
    print(f"Vector store built with {len(chunks)} vectors")

    query = "What is a graph neural network? What are the applications of graph neural networks?"
    results = search(query, client, model)

    for i, result in enumerate(results):
        print(f"\nResult {i+1} (score: {result.score:.3f}):")
        print(result.payload["text"])
        print(f"Page: {result.payload['page']}")