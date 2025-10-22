import numpy as np
import pandas as pd
import unicodedata

# Importamos después de tener definidas las utilidades
from app.nlp_core import FAQSearch

def _strip_accents(s: str) -> str:
    """Quita acentos/diacríticos para que el match por substring sea robusto."""
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

class DummyEncoder:
    """
    Simula el comportamiento de SentenceTransformer.encode() devolviendo
    embeddings unitarios e1/e2/e3 según palabras clave en la consulta.
    - e1: inscrip/inscrib/anot → inscripciones
    - e2: matric/pago          → matrícula
    - e3: else                 → modalidad
    """
    def __init__(self, e1, e2, e3):
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def encode(self, texts, normalize_embeddings=True):
        outs = []
        for t in texts:
            t_low = t.lower()
            t_norm = _strip_accents(t_low)
            if ("inscrip" in t_norm) or ("inscrib" in t_norm) or ("anot" in t_norm):
                outs.append(self.e1)
            elif ("matric" in t_norm) or ("pago" in t_norm):
                outs.append(self.e2)
            else:
                outs.append(self.e3)
        return np.vstack(outs).astype(np.float32)

def test_faqsearch_query_with_synthetic_data(tmp_path, monkeypatch):
    """
    Crea un CSV y embeddings sintéticos, monkeypatchea el encoder para NO descargar modelos,
    y valida que FAQSearch.query() devuelva la respuesta correcta por similitud.
    """
    # 1) Armar data/paths temporales
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    faqs_path = data_dir / "faqs.csv"
    emb_path = data_dir / "embeddings.npz"

    # 2) CSV sintético de FAQs
    df = pd.DataFrame({
        "pregunta_faq": [
            "¿Cuándo son las inscripciones?",     # idx 0
            "¿Dónde pago la matrícula?",          # idx 1
            "¿Cuál es la modalidad de cursado?"   # idx 2
        ],
        "respuesta": [
            "Las inscripciones abren en agosto y son online.",
            "La matrícula se paga en sede o vía portal de pagos.",
            "Podés cursar presencial o a distancia según la carrera."
        ]
    })
    df.to_csv(faqs_path, index=False)

    # 3) Embeddings sintéticos normalizados (ejes ortogonales)
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # inscripciones
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # matrícula
    e3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # modalidad
    emb = np.vstack([e1, e2, e3])
    np.savez(emb_path, vectors=emb)

    # 4) Monkeypatch: reemplazar SentenceTransformer por DummyEncoder
    import app.nlp_core as nlp_core_mod
    monkeypatch.setattr(
        nlp_core_mod,
        "SentenceTransformer",
        lambda *a, **k: DummyEncoder(e1, e2, e3)
    )

    # 5) Instanciar con paths temporales y threshold razonable
    faq = FAQSearch(
        faqs_path=str(faqs_path),
        emb_path=str(emb_path),
        model_name="IGNORED_BY_DUMMY",
        threshold=0.70
    )

    # 6) Consultas y aserciones (verificamos índices esperados)
    q1 = "¿Desde cuándo me puedo inscribir?"     # debe mapear a idx 0 (inscripciones)
    ans1, score1, idx1 = faq.query(q1)
    assert ans1 is not None
    assert idx1 == 0
    assert np.isclose(score1, 1.0)

    q2 = "Cómo pagar la matrícula"               # debe mapear a idx 1 (matrícula)
    ans2, score2, idx2 = faq.query(q2)
    assert ans2 is not None
    assert idx2 == 1
    assert np.isclose(score2, 1.0)

    q3 = "¿Qué modalidad hay?"                   # debe mapear a idx 2 (modalidad)
    ans3, score3, idx3 = faq.query(q3)
    assert ans3 is not None
    assert idx3 == 2
    assert np.isclose(score3, 1.0)