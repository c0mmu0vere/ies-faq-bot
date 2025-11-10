# app/retriever.py

import faiss
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

# ===== Rutas a artefactos =====
FAISS_INDEX_PATH = "models/embeddings_index.faiss"
FAQS_PICKLE_PATH = "models/faqs.pkl"

# ===== Carga de modelo denso y recursos FAISS =====
# Modelo multilingüe (ya lo venías usando)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Índice FAISS (IndexFlatIP) con vectores normalizados
index = faiss.read_index(FAISS_INDEX_PATH)

# === NUEVO: extraer la matriz de embeddings del índice (IndexFlatIP) ===
try:
    _XB = faiss.vector_to_array(index.xb).reshape(index.ntotal, index.d).astype(np.float32, copy=False)
except Exception:
    _XB = None

with open(FAQS_PICKLE_PATH, "rb") as f:
    faqs: List[Dict] = pickle.load(f)

# ====== TF-IDF (índice léxico) ======
# Construimos un índice léxico sobre las preguntas, para complementar el denso
# Requiere scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    _HAS_SK = True
except Exception:
    _HAS_SK = False
    TfidfVectorizer = None
    linear_kernel = None

_tfidf_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix = None
_faq_texts: Optional[List[str]] = None


def _build_sparse_index(faqs_list: List[Dict]) -> None:
    """Construye el índice TF-IDF sobre las preguntas de las FAQs."""
    global _tfidf_vectorizer, _tfidf_matrix, _faq_texts
    if not _HAS_SK:
        # Si no está sklearn disponible, dejamos el híbrido desactivado
        _tfidf_vectorizer = None
        _tfidf_matrix = None
        _faq_texts = None
        return

    _faq_texts = [f'{f["pregunta_faq"]}  {f["respuesta"]}' for f in faqs_list]
    _tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,  # ayuda a separar bien términos como "modalidad" vs "salida laboral"
        ngram_range=(1, 2),    # capta bigramas útiles
        sublinear_tf=True,
        max_features=60000
    )
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(_faq_texts)


# Construimos el índice léxico una sola vez al importar
_build_sparse_index(faqs)


# ===== Helpers comunes =====
def encode_query(query: str) -> np.ndarray:
    """
    Codifica y normaliza la consulta para obtener su vector de embeddings.
    Retorna float32 con norma 1 (FAISS IP ≈ coseno).
    """
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return vec.astype(np.float32, copy=False)


def _dense_topk(query_vec: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    """
    Top-k sobre FAISS (IP con embeddings normalizados).
    Retorna lista [(idx, score_cos)] con índices de faqs.
    """
    if query_vec.dtype != np.float32:
        query_vec = query_vec.astype(np.float32, copy=False)
    D, I = index.search(query_vec, k)
    # Filtramos -1 por seguridad (no debería aparecer con IndexFlatIP)
    return [(int(I[0][i]), float(D[0][i])) for i in range(I.shape[1]) if I[0][i] != -1]


def _sparse_topk(query_text: str, k: int = 10) -> List[Tuple[int, float]]:
    """
    Top-k TF-IDF (coseno). Retorna lista [(idx, score_lex)].
    Si sklearn no está disponible o no hay texto, retorna [].
    """
    if not _HAS_SK or _tfidf_vectorizer is None or not query_text or not query_text.strip():
        return []
    q = _tfidf_vectorizer.transform([query_text])
    sims = linear_kernel(q, _tfidf_matrix).ravel()  # similitud coseno con la matriz TF-IDF
    top_idx = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in top_idx]

def _rrf(
    dense: List[Tuple[int, float]],
    sparse: List[Tuple[int, float]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF).
    Toma rankings de 2 listas [(idx, score)] y combina por posición:
        RRF = 1/(k + rank_dense) + 1/(k + rank_sparse)
    Devuelve [(idx, rrf_score)] ordenado desc.
    """
    # rank por canal (1 = mejor)
    d_sorted = sorted(dense, key=lambda x: x[1], reverse=True)
    s_sorted = sorted(sparse, key=lambda x: x[1], reverse=True)
    d_rank = {i: r for r, (i, _) in enumerate(d_sorted, start=1)}
    s_rank = {i: r for r, (i, _) in enumerate(s_sorted, start=1)}

    all_ids = set(d_rank) | set(s_rank)
    fused: List[Tuple[int, float]] = []
    for i in all_ids:
      rd = d_rank.get(i, 10_000)
      rs = s_rank.get(i, 10_000)
      fused.append((i, 1.0 / (k + rd) + 1.0 / (k + rs)))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def _minmax_norm(items: List[Tuple[int, float]]) -> Dict[int, float]:
    """
    Normaliza scores a [0,1] por min-max local.
    items: lista [(idx, score)]
    """
    if not items:
        return {}
    scores = [s for _, s in items]
    vmin, vmax = min(scores), max(scores)
    denom = (vmax - vmin) if (vmax - vmin) > 1e-8 else 1.0
    return {i: (s - vmin) / denom for i, s in items}


def _fuse_scores(
    dense: List[Tuple[int, float]],
    sparse: List[Tuple[int, float]],
    alpha: float = 0.6
) -> List[Tuple[int, float]]:
    """
    Fusión simple por suma ponderada:
    score_fused = alpha * score_denso_norm + (1 - alpha) * score_lex_norm

    - Normalizamos cada canal a [0,1] sobre sus TOP locales
    - Retorna lista [(idx, score_fused)] ordenada desc.
    """
    d_norm = _minmax_norm(dense)
    s_norm = _minmax_norm(sparse)

    all_ids = set([i for i, _ in dense]) | set([i for i, _ in sparse])
    fused: List[Tuple[int, float]] = []
    for i in all_ids:
        d_sc = d_norm.get(i, 0.0)
        s_sc = s_norm.get(i, 0.0)
        fused_sc = alpha * d_sc + (1.0 - alpha) * s_sc
        fused.append((i, fused_sc))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# ===== API principal =====
def buscar_similares(query_vec: np.ndarray, top_k: int = 5, query_text: Optional[str] = None) -> List[Dict]:
    """
    Recuperación híbrida: denso (FAISS) + léxico (TF-IDF).
    - query_vec: vector normalizado (norma 1, float32)
    - query_text: texto crudo de la consulta (para TF-IDF)
    """
    # 1) denso (pedimos MÁS que top_k para ampliar el recall en la fusión)
    dense_k = max(top_k, 50)  # <-- AUMENTADO (antes 10)
    dense = _dense_topk(query_vec, k=dense_k)  # [(idx, cos_denso)]

    # Si no hay texto o no se pudo construir el índice léxico, mantenemos solo denso
    if not query_text or not query_text.strip() or not _HAS_SK or _tfidf_vectorizer is None:
        resultados = []
        for idx, sc in dense[:top_k]:
            faq = faqs[idx]
            resultados.append({
                "faq_id": faq["faq_id"],
                "pregunta_faq": faq["pregunta_faq"],
                "respuesta": faq["respuesta"],
                "score": float(sc),
                "score_dense": float(sc),
                "score_lex": 0.0,
                "score_fused": float(sc),  # para mantener estructura de debug
            })
        return resultados

    # 2) léxico (mismo K ampliado)
    q_lower = (query_text or "").lower()
    laboral_hint = any(k in q_lower for k in [
    "salida", "salidas", "laboral", "trabajo", "trabajar",
    "áreas", "ambitos", "ámbitos", "egresad", "oportunidades",
    "campos", "campo laboral", "puestos", "empleo", "empleabilidad",
    "recursos humanos", "rrhh"])

    expanded_text = query_text
    if laboral_hint:
        expanded_text = (
            query_text
            + " salida laboral salidas laborales ámbitos de trabajo campos laborales "
            "puestos empleabilidad empleo roles tareas gestión de personas recursos humanos rrhh "
            "competencias perfil egreso"
        )

    sparse_k = max(top_k, 50)  # <-- AUMENTADO (antes 10)
    sparse = _sparse_topk(expanded_text, k=sparse_k)  # [(idx, cos_lex)]

    # 3) fusión (RRF + penalización contextual si aplica)
    fused = _rrf(dense, sparse, k=60)

    if laboral_hint:
        adjusted = []
        for idx, sc in fused:
            title = faqs[idx]["pregunta_faq"].lower()
            text  = (faqs[idx]["pregunta_faq"] + " " + faqs[idx]["respuesta"]).lower()

            if ("modalidad" in title) or ("cursar" in title):
                sc -= 0.30
            if ("dura" in title) or ("duración" in title):
                sc -= 0.20
            if ("título" in title) or ("otorga" in title):
                sc -= 0.15
            if ("laboratorio" in text) or ("informática" in text):
                sc -= 0.25

            if ("ámbitos" in title) or ("trabajar" in title) or ("salida laboral" in title):
                sc += 0.15

            if ("práctica" in text) or ("practica" in text) or ("profesionalizante" in text):
                sc -= 0.20

            adjusted.append((idx, sc))
        fused = sorted(adjusted, key=lambda x: x[1], reverse=True)

    # 4) armar salida (orden híbrido) calculando siempre score_dense real
    resultados = []
    qv = query_vec[0]  # (d,) float32 normalizado
    for idx, fused_sc in fused[:top_k]:
        faq = faqs[idx]
        d_sc_list = [s for i, s in dense if i == idx]
        if d_sc_list:
            d_sc = float(d_sc_list[0])
        elif _XB is not None:
            d_sc = float(np.dot(qv, _XB[idx]))  # coseno real
        else:
            d_sc = 0.0

        s_sc = next((s for i, s in sparse if i == idx), 0.0)

        resultados.append({
            "faq_id": faq["faq_id"],
            "pregunta_faq": faq["pregunta_faq"],
            "respuesta": faq["respuesta"],
            "score": float(d_sc),          # el selector usa este (coseno denso)
            "score_dense": float(d_sc),
            "score_lex": float(s_sc),
            "score_fused": float(fused_sc)
        })
    return resultados