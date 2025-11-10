# scripts/build_index.py

import os
import pickle
import faiss
import numpy as np
from app.utils import load_faqs
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Ruta del archivo CSV de FAQs
FAQ_PATH = "data/faqs.csv"
# Directorio de salida
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Cargar FAQs
faqs = load_faqs(FAQ_PATH)
print(f"Se cargaron {len(faqs)} FAQs.")
faq_texts = [faq["pregunta_faq"] for faq in faqs]
print(f"Se cargaron {len(faq_texts)} preguntas para indexar.")

# 2. Generar embeddings
print("Cargando modelo de embeddings...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(
    faq_texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False  # ← Normalización manual más abajo
)

# 3. Normalizar embeddings (para similitud coseno con IP)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# 4. Crear índice FAISS con similitud coseno
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # IP = inner product (para similitud coseno si los vectores están normalizados)
index.add(embeddings)
print(f"Índice FAISS creado con {index.ntotal} vectores.")

# 5. Guardar índice y preguntas
faiss.write_index(index, os.path.join(MODEL_DIR, "embeddings_index.faiss"))
with open(os.path.join(MODEL_DIR, "faqs.pkl"), "wb") as f:
    pickle.dump(faqs, f)

print("Embeddings e índice FAISS guardados con éxito.")