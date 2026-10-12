# RAG Chatbot (local Ollama)

This folder contains a minimal retrieval-augmented generation (RAG) example that uses:

- SentenceTransformers to embed documents
- FAISS for vector search
- Local Ollama CLI to run a local LLM for generation (you must install Ollama separately)

Files:

- `server.py` - Flask app exposing `/chat` and `/reload` endpoints and a simple web UI at `/`.
- `requirements.txt` - Python dependencies for the RAG service.
- `docs/` - Add `.txt` files here to include them in the retrieval corpus.

Quick start (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r RAG/requirements.txt
# Install Ollama separately: https://ollama.com
# Ensure `ollama` CLI is on PATH and you have a model installed, e.g. `ollama pull llama2`
python RAG/server.py
```

Then open http://localhost:5001 in your browser.
