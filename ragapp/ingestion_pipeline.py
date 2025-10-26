import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


def load_documents(docs_path="pdf_docs"):
    """Load all PDF files from the pdf_docs directory"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your company files.")

    loader = DirectoryLoader(
        docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader  # type: ignore
    )

    documents = loader.load()

    # print(f"Loaded {len(documents)} document(s)")
    if len(documents) == 0:
        raise FileNotFoundError(
            f"The directory {docs_path} does not contain any PDF files.")

    for i, doc in enumerate(documents[:1]):
        print(f"\n Document {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        # print(f"  metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
        else:
            print(f"\n... Has only less chunks")

    return chunks


def download_ollama_face_embedding():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    return embeddings


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embeddings = download_ollama_face_embedding()

    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    """ Main ingestion pipeline"""

    print("Running ingestion pipeline...")

    documents = load_documents(docs_path="pdf_docs")

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks)
