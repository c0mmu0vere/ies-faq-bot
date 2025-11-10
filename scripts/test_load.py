# scripts/test_load.py

from app.utils import load_faqs

if __name__ == "__main__":
    faqs = load_faqs("data/faqs.csv")
    print(f"Se cargaron {len(faqs)} FAQs.")
    print(faqs[0])  # Mostrar la primera para inspección