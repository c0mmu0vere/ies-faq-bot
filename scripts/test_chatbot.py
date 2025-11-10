# scripts/test_chatbot.py
from app.retriever import encode_query, buscar_similares
from app.response_selector import seleccionar_respuesta, SelectorConfig

def main():
    cfg = SelectorConfig(
        tau_high=0.80,      # alto → extractivo
        tau_low=0.55,       # medio → generativo / tie-break
        near_tie_delta=0.05 # empate cercano → desambiguar
    )

    print("Test del selector (Etapa 6). Escribí una consulta o 'salir' para terminar.\n")

    while True:
        try:
            q = input("Consulta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFin.")
            break

        if q.lower() in {"salir", "exit", "quit"}:
            print("Fin.")
            break
        if not q:
            continue

        try:
            q_vec = encode_query(q)                        # norma 1, float32
            candidatos = buscar_similares(q_vec, top_k=5, query_text=q)  # [{faq_id, pregunta_faq, respuesta, score}]
            print(candidatos[0])

            out = seleccionar_respuesta(
                query=q,
                candidatos=candidatos,
                cfg=cfg,
                enable_generation=True  # por ahora sin LLM (Etapa 7)
            )

            print("\n--- RESULTADO ---")
            print(f"[modo] {out['mode']}")
            print(f"[respuesta]\n{out['answer']}\n")
            meta = out.get("meta", {})
            best = meta.get("best_score")
            if best is not None:
                print(f"[meta] best={best:.4f}  top1={meta.get('top1_question')}")
            print("-----------------\n")

        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}\n")

if __name__ == "__main__":
    main()