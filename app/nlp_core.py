import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def cosine_similarity_manual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (n,d) normalizado | b: (m,d) normalizado → retorna (n,m) con cosenos (= dot product)
    return np.dot(a, b.T)

class FAQSearch:
    def __init__(
        self,
        faqs_path: str = "data/faqs.csv",
        emb_path: str = "data/embeddings.npz",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        threshold: float = 0.60,
    ):
        self.df = pd.read_csv(faqs_path)
        self.emb = np.load(emb_path)["vectors"]  # (N,d)
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def query(self, text: str):
        q_vec = self.model.encode([text], normalize_embeddings=True)  # (1,d)
        sims = cosine_similarity_manual(q_vec, self.emb)[0]  # (N,)
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= self.threshold:
            return self.df.iloc[idx]["respuesta"], score, idx
        return None, score, idx  # activa fallback fuera