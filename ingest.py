import os
import shutil
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


CHROMA_PATH = "chroma"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(
    PROJECT_ROOT,
    "RAGProject",
    "Data",
    "Optimizing Redundancy Detection in Software Requirement Specifications Using BERT Embeddings.pdf"
)


def ensure_api_key_present() -> None:
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Create a .env with GOOGLE_API_KEY=..."
        )


def load_pdf_documents(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # Attach a consistent source for later citation
    for doc in documents:
        doc.metadata = {**doc.metadata, "source": pdf_path}
    return documents


def split_text_into_chunks(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_chunks_to_chroma(chunks: List[Document]) -> None:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def main() -> None:
    ensure_api_key_present()
    docs = load_pdf_documents(PDF_PATH)
    chunks = split_text_into_chunks(docs)
    save_chunks_to_chroma(chunks)


if __name__ == "__main__":
    main()



