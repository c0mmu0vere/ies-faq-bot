# scripts/debug_embeddings.py

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Cargamos las FAQs
with open("models/faqs.pkl", "rb") as f:
    faqs = pickle.load(f)

# Mostramos la primera pregunta
print(f"Primera pregunta: {faqs[0]['pregunta_faq']}")

# Recalculamos el embedding desde cero, normalizado
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
emb = model.encode([faqs[0]['pregunta_faq']], normalize_embeddings=True)
print(f"Embedding normalizado (primeros 5 valores): {emb[0][:5]}")
print(f"Norma del embedding: {np.linalg.norm(emb[0]):.4f}")
