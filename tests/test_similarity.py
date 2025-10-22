import numpy as np
from app.nlp_core import cosine_similarity_manual

def test_cosine_manual_shapes():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2
    b = np.array([[1.0, 0.0]])              # 1x2
    sim = cosine_similarity_manual(a, b)    # 2x1
    assert sim.shape == (2, 1)
    # si están normalizados, cosenos esperados: [1.0, 0.0]
    assert np.isclose(sim[0,0], 1.0, atol=1e-6)
    assert np.isclose(sim[1,0], 0.0, atol=1e-6)