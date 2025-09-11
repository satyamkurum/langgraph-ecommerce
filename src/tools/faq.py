# src/tools/faq.py
import json
import os
from typing import List, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_EMBED = True
except Exception:
    HAS_EMBED = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faq.json")

class FAQRetriever:
    def __init__(self, faq_path: str = FAQ_PATH, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faq = json.load(f)  # dict: question_key -> answer
        self.keys = list(self.faq.keys())
        self.answers = [self.faq[k] for k in self.keys]

        if HAS_EMBED:
            self.embed_model = SentenceTransformer(model_name)
            self._build_faiss()
        else:
            self.embed_model = None
            self.index = None

    def _build_faiss(self):
        texts = self.keys
        embs = self.embed_model.encode(texts, convert_to_numpy=True)
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        # normalize
        import numpy as np
        faiss.normalize_L2(embs)
        self.index.add(embs)

    def lookup_simple(self, query: str) -> str:
        q = query.lower()
        for k, v in self.faq.items():
            if k in q:
                return v
        return "I don't have an exact FAQ match for that."

    def lookup_semantic(self, query: str, top_k: int = 1) -> Tuple[str, float]:
        """Return best matching answer and score (0-1)."""
        if not HAS_EMBED or self.index is None:
            return self.lookup_simple(query), 0.0
        import numpy as np
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        idx = int(I[0, 0])
        score = float(D[0, 0])
        return self.answers[idx], score

# convenience
_default_retriever: Optional[FAQRetriever] = None

def get_faq_retriever() -> FAQRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = FAQRetriever()
    return _default_retriever

def faq_lookup(query: str) -> str:
    r = get_faq_retriever()
    ans, score = r.lookup_semantic(query)
    # conservative threshold
    if score < 0.5:
        # fallback to simple lookup
        return r.lookup_simple(query)
    return ans
