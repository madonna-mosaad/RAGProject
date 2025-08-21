## RAG over PDF using Gemini (Google Generative AI)

This project builds a simple Retrieval-Augmented Generation (RAG) system over the paper in `Data/Optimizing Redundancy Detection in Software Requirement Specifications Using BERT Embeddings.pdf`.

- **Embeddings**: `text-embedding-004` via Google Generative AI
- **Vector store**: Chroma (local persistence in `chroma/`)
- **LLM**: Gemini 1.5 Flash via Google Generative AI

### 1) Setup

1. Install Python 3.10+.
2. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
python -m pip install -r RAGProject\requirements.txt
```

3. Configure environment variables:

```powershell
# Edit .env and set GOOGLE_API_KEY, or set for this session only:
$env:GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
```

### 2) Ingest the PDF into Chroma

```powershell
.\.venv\Scripts\python RAGProject\ingest.py
```

This reads the PDF from `Data/`, chunks it, computes Gemini embeddings, and writes them to `chroma/`.

### 3) Ask questions
Open the Terminal in PyCharm (bottom tab)

Type this command:
```powershell
python query.py "What is redundancy detection in software requirements?"
```


You will get an answer grounded in the most relevant chunks plus the source path(s).


### Notes

- If you re-run ingestion, the `chroma/` directory is cleared and rebuilt.
- Ensure your PDF is present at `Data/Optimizing Redundancy Detection in Software Requirement Specifications Using BERT Embeddings.pdf`.
- Uses only Google Generative AI (Gemini) — no OpenAI credentials required.


