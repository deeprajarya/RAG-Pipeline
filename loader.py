# need to load the pdf file and split the text

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,                      # chunk size is the number of characters in each chunk
        chunk_overlap=50                     # chunk overlap is the number of characters to overlap between chunks
    )
    chunks = splitter.split_documents(pages)

    return chunks

if __name__ == "__main__":
    pdf_path = "data/Introduction to Graph Neural Networks.pdf"
    chunks = load_and_chunk(pdf_path)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0].page_content}")
    print(f"Last chunk: {chunks[-1].page_content}")
    print(f"Metadata: {chunks[0].metadata}")