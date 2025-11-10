import os
import sys
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Ajustar path para imports relativos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils import load_faqs

# Cargar preguntas y sus datos originales
faqs = load_faqs("data/faqs.csv")
preguntas = [f['pregunta_faq'] for f in faqs]

# Cargar embeddings y FAQ originales
with open("models/faqs.pkl", "rb") as f:
    original_faqs = pickle.load(f)

# Cargar modelo y normalizar
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.read_index("models/embeddings_index.faiss")

print("Escribí una consulta (o 'salir' para terminar):")
while True:
    query = input("Consulta: ").strip()
    if query.lower() in ['salir', 'exit', 'quit']:
        break

    # Codificar y normalizar el vector de consulta
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec, k=5)

    print("\n🔎 Top 5 resultados más similares:")
    for rank, idx in enumerate(I[0]):
        score = D[0][rank]  # Similitud coseno real (ya normalizados)
        faq = original_faqs[idx]
        print(f"{rank+1}. Score: {score:.4f}")
        print(f"   Pregunta: {faq['pregunta_faq']}")
        print(f"   Respuesta: {faq['respuesta'][:150]}...\n")
    print("-" * 70)