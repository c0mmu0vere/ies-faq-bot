from app.nlp_core import FAQSearch

faq = FAQSearch(threshold=0.72)
while True:
    q = input("\nTu pregunta (ENTER para salir): ").strip()
    if not q:
        break
    ans, score, idx = faq.query(q)
    if ans is None:
        print(f"[Fallback] No encontré una coincidencia confiable (score={score:.3f}).")
    else:
        print(f"[Match {score:.3f}] → {ans}")