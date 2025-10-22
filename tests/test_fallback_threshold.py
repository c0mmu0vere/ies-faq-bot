import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from app.nlp_core import FAQSearch

def test_fallback_when_below_threshold(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    faqs_path = data_dir / "faqs.csv"
    emb_path = data_dir / "embeddings.npz"

    df = pd.DataFrame({
        "pregunta_faq": ["A", "B"],
        "respuesta": ["resp A", "resp B"]
    })
    df.to_csv(faqs_path, index=False)

    e1 = np.array([1.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0], dtype=np.float32)
    np.savez(emb_path, vectors=np.vstack([e1, e2]))

    class DummyEncoder:
        def encode(self, texts, normalize_embeddings=True):
            # Devolvemos un vector equidistante (0.707 con ambas si normalizamos)
            v = (e1 + e2) / np.linalg.norm(e1 + e2)
            return np.vstack([v for _ in texts]).astype(np.float32)

    import app.nlp_core as nlp_core_mod
    monkeypatch.setattr(nlp_core_mod, "SentenceTransformer", lambda *a, **k: DummyEncoder())

    # Umbral alto para forzar fallback (cosenos ~0.707)
    faq = FAQSearch(
        faqs_path=str(faqs_path),
        emb_path=str(emb_path),
        model_name="IGNORED",
        threshold=0.90
    )

    ans, score, idx = faq.query("consulta ambigua")
    assert ans is None
    assert score < 0.90
