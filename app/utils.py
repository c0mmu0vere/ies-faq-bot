# app/utils.py
import pandas as pd
import csv

def load_faqs(path):
    faqs = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pregunta = row.get('pregunta_faq', '').strip()
            respuesta = row.get('respuesta', '').strip()

            if pregunta and respuesta and len(pregunta) > 5:
                faqs.append({
                    'faq_id': str(i),
                    'pregunta_faq': pregunta,
                    'respuesta': respuesta
                })
            else:
                print(f"[LÍNEA OMITIDA] {i}: Pregunta o respuesta vacía")
    
    print(f"Se cargaron {len(faqs)} FAQs.")
    return faqs

