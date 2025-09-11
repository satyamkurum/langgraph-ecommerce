# src/tools/recommender.py
import json
import os
from typing import List, Dict
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_EMBED = True
except Exception:
    HAS_EMBED = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "products.json")

class SimpleRecommender:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
            self.products = json.load(f)
        self.texts = [p["title"] + ". " + p.get("description","") for p in self.products]
        if HAS_EMBED:
            self.model = SentenceTransformer(model_name)
            self._build_index()
        else:
            self.model = None
            self.index = None

    def _build_index(self):
        import numpy as np
        embs = self.model.encode(self.texts, convert_to_numpy=True)
        faiss.normalize_L2(embs)
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embs)

    def recommend(self, query: str, top_k: int = 3) -> List[Dict]:
        if not HAS_EMBED:
            # fallback: simple keyword match
            q = query.lower()
            out = []
            for p in self.products:
                if any(tok in (p["title"]+p.get("description","")).lower() for tok in q.split()):
                    out.append(p)
            return out[:top_k]
        import numpy as np
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        ids = [int(i) for i in I[0].tolist()]
        return [self.products[i] for i in ids]

# convenience
_rec = None
def get_recommender():
    global _rec
    if _rec is None:
        _rec = SimpleRecommender()
    return _rec

def recommend_products(query: str, top_k: int = 3):
    rec = get_recommender()
    return rec.recommend(query, top_k)
