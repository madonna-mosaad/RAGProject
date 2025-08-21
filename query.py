import argparse
import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context. If the answer is not present in the context, say you don't know.

Context:
{context}

---

Question: {question}
"""


def ensure_api_key_present() -> None:
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Create a .env with GOOGLE_API_KEY=...")
    genai.configure(api_key=api_key)


def get_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db


def build_context(results: List[Tuple]) -> str:
    return "\n\n---\n\n".join([doc.page_content for doc, _score in results])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--min_relevance", type=float, default=0.5, help="Min relevance threshold (0-1)")
    args = parser.parse_args()

    ensure_api_key_present()
    db = get_vectorstore()

    results = db.similarity_search_with_relevance_scores(args.query_text, k=args.k)
    if len(results) == 0 or (results and results[0][1] < args.min_relevance):
        print("Unable to find matching results.")
        return

    context_text = build_context(results)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=args.query_text)

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    response_text = response.text if hasattr(response, "text") else str(response)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()



