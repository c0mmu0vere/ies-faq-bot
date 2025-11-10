# scripts/test_api.py
import requests
import json

URL = "http://127.0.0.1:8000/chat"

def ask(q):
    payload = {"query": q, "top_k": 5, "enable_generation": True}
    r = requests.post(URL, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    print("\nQ:", q)
    print("mode:", data.get("mode"))
    print("answer:\n", data.get("answer"))
    print("meta.best_dense:", data.get("meta", {}).get("best_dense"))
    print("meta.generator_backend:", data.get("meta", {}).get("generator_backend"))
    return data

if __name__ == "__main__":
    ask("¿En qué ámbitos puede trabajar un Técnico Superior en Recursos Humanos?")
    ask("¿Qué oportunidades laborales tiene la carrera?")
