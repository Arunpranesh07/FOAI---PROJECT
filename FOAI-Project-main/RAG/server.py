import os
import glob
import subprocess
import sys
from typing import List, Tuple, Optional

from flask import Flask, request, jsonify, render_template, Response
from sentence_transformers import SentenceTransformer
import numpy as np

# Ensure stdout/stderr use utf-8 (helps when printing emoji on some platforms)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import faiss
except Exception:
    faiss = None


class RAGChat:
    def __init__(self, docs_dir: str = None, model_name: str = "all-MiniLM-L6-v2"):
        base = os.path.dirname(__file__)
        self.docs_dir = docs_dir or os.path.join(base, "docs")
        self.model_name = model_name

        # Default to Llama 3.2 (can be overridden by OLLAMA_MODEL env var)
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.ollama_cli = os.getenv("OLLAMA_CLI", "ollama")

        # sentence-transformers model for retrieval
        self.model = SentenceTransformer(self.model_name)

        self.texts: List[str] = []
        self.metadata: List[dict] = []
        self.index = None
        self.embeddings = None

        self._load_and_index()

    def _load_and_index(self):
        # load text files (utf-8)
        files = sorted(glob.glob(os.path.join(self.docs_dir, "**", "*.txt"), recursive=True))
        if not files:
            os.makedirs(self.docs_dir, exist_ok=True)

        self.texts = []
        self.metadata = []
        for p in files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                if txt:
                    self.texts.append(txt)
                    self.metadata.append({"path": os.path.relpath(p, start=os.path.dirname(__file__))})
            except Exception as e:
                # skip bad files but print for debug
                print(f"Failed reading {p}: {e}")

        if not self.texts:
            self.index = None
            return

        # compute embeddings
        emb = self.model.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = emb.astype("float32")

        # build FAISS index if available
        if faiss is None:
            print("Warning: faiss not available — retrieval disabled. Install faiss-cpu for faster retrieval.")
            self.index = None
            return

        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings)
        self.index = index
        print(f"Indexed {len(self.texts)} documents.")

    def reload(self):
        self._load_and_index()

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[float, str, dict]]:
        """Return list of (score, text, metadata). If faiss not present, returns empty list."""
        if self.index is None:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        results: List[Tuple[float, str, dict]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append((float(score), self.texts[idx], self.metadata[idx]))
        return results

    def generate_with_ollama(self, prompt: str, max_tokens: int = 512, timeout: int = 60) -> str:
        """
        Generate response using Ollama CLI with proper command structure.
        Returns the model output as a string or a helpful error message.
        """
        try:
            # Use the correct Ollama CLI command structure
            cmd = [self.ollama_cli, "run", self.ollama_model]
            
            # Run the command with the prompt as stdin
            proc = subprocess.run(
                cmd, 
                input=prompt, 
                capture_output=True, 
                text=True, 
                check=False,  # Don't raise on non-zero exit
                timeout=timeout,
                encoding='utf-8'
            )
            
            # Get output from stdout
            stdout = proc.stdout.strip() if proc.stdout else ""
            stderr = proc.stderr.strip() if proc.stderr else ""
            
            # If we got output from stdout, return it
            if stdout:
                return stdout
            
            # If no stdout but stderr has content, it might be an error
            if stderr:
                print(f"Ollama stderr: {stderr}")
                return f"ERROR: {stderr}"
            
            # If process failed with no output
            if proc.returncode != 0:
                return f"ERROR: Ollama process failed with exit code {proc.returncode}"
            
            # Empty response
            return "ERROR: No response from model"
            
        except FileNotFoundError:
            return (
                "ERROR: Ollama CLI not found. Install Ollama (https://ollama.com) and ensure 'ollama' is on PATH."
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: Ollama request timed out after {timeout} seconds"
        except Exception as e:
            return f"ERROR: Unexpected error: {str(e)}"

    # Optional: small emoji post-processor if model doesn't include emoji
    @staticmethod
    def add_emojis(text: str) -> str:
        emoji_map = {
            "hello": "👋",
            "hi": "👋",
            "thank you": "🙏",
            "thankyou": "🙏",
            "good morning": "🌞",
            "i love you": "❤️",
            "love": "❤️",
            "yes": "👍",
            "no": "❌",
            "school": "🏫",
            "sorry": "🙏",
            "please": "🙏",
            "done": "✅",
            "opposite": "↩️",
            "reverse": "↩️",
            "great": "🎉",
            "thanks": "🙏",
            "cool": "😎",
            "idea": "💡",
            "question": "❓",
        }
        lower = text.lower()
        # only add emojis if none present (quick heuristic)
        if any(ord(ch) > 10000 for ch in text):
            return text
        for k, v in emoji_map.items():
            if k in lower:
                # Replace occurrences case-insensitively while preserving original casing where possible
                text = text.replace(k, f"{k} {v}")
                # also do title-case variant in case original text uses it
                text = text.replace(k.title(), f"{k.title()} {v}")
        return text


# Flask app
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
rag: Optional[RAGChat] = None


def _ensure_rag_initialized():
    global rag
    if rag is None:
        base = os.path.dirname(__file__)
        docs_dir = os.path.join(base, "docs")
        rag = RAGChat(docs_dir=docs_dir)


@app.before_request
def _ensure_rag_before_request():
    _ensure_rag_initialized()


@app.after_request
def set_utf8(response: Response):
    """
    Ensure responses declare UTF-8 charset so browsers render emojis correctly.
    """
    ctype = response.headers.get("Content-Type", "")
    if "application/json" in ctype:
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    elif "text/html" in ctype or ctype == "":
        # default for templates
        response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        top_k = int(data.get("top_k", 4))
        if not message:
            return jsonify({"error": "Missing 'message' in request"}), 400

        print(f"Received message: {message}")

        # Retrieval
        retrieved = rag.retrieve(message, top_k=top_k) if rag and rag.index is not None else []
        if retrieved:
            contexts = "\n\n".join([f"Source ({r[2].get('path')}):\n{r[1]}" for r in retrieved])
            print(f"Retrieved {len(retrieved)} documents")
        else:
            contexts = "No context available."
            print("No documents retrieved")

        # Build prompt (ask model to include emojis)
        prompt = (
            "You are an assistant that understands sign-language descriptions and maps them to clear, natural-language answers. "
            "Use the provided CONTEXT to answer when relevant. If the CONTEXT does not contain relevant information, "
            "you can answer based on general knowledge in a friendly and concise manner. "
            "Add appropriate emojis in your short answers to make them lively.\n\n"
            f"CONTEXT:\n{contexts}\n\nQUESTION:\n{message}\n\nANSWER:"
        )

        print("Generating response with Ollama...")
        
        # Generate answer via Ollama CLI
        raw_answer = rag.generate_with_ollama(prompt)
        
        print(f"Raw answer: {raw_answer[:100]}...")

        # If Ollama failed or returned error string, fall back to a safe reply
        if raw_answer.startswith("ERROR:") or raw_answer.strip() == "":
            print(f"Ollama error: {raw_answer}")
            fallback = (
                "Sorry — I couldn't generate a response using the local model. "
                "Try again or check the server logs. 🤖"
            )
            # Allow adding some emoji flair
            answer = RAGChat.add_emojis(fallback)
        else:
            # raw_answer may include multiple lines; strip safely
            answer = raw_answer.strip()
            # If the model didn't include emojis, add simple ones for common phrases
            if not any(ord(ch) > 10000 for ch in answer):  # rough check for existing emoji chars
                answer = RAGChat.add_emojis(answer)

        sources = [r[2] for r in retrieved] if retrieved else []
        return jsonify({"answer": answer, "sources": sources})
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/reload", methods=["POST"])
def reload_index():
    rag.reload()
    return jsonify({"status": "reloaded", "documents": len(rag.texts)})


if __name__ == "__main__":
    # Run dev server
    # Note: production deployments should use a proper WSGI server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=True)