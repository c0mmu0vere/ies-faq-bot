# IES FAQ Chatbot — Documentación Técnica (v0.2)

## 🚀 Descripción General

Este proyecto implementa un Chatbot basado en Inteligencia Artificial para responder preguntas frecuentes sobre carreras, materias, modalidades, trámites y servicios del IES. La arquitectura está diseñada para combinar *recuperación semántica* y *recuperación léxica* con un *selector inteligente* que decide cuándo usar respuestas directas, cuándo pedir aclaraciones y cuándo reformular usando un modelo generativo (LLM).

---

## 📦 Características Principales

- Recuperación híbrida:
  - Embeddings densos (FAISS + transformer multilingüe)
  - Recuperación léxica (TF-IDF)
  - Fusión por RRF (Reciprocal Rank Fusion)
- Selector de respuesta inteligente:
  - Extractive
  - Generative (polish)
  - Clarify (tie-break)
  - Fallback
- Integración con LLM:
  - Ollama (local) con Llama 3
  - Alternativa: OpenAI GPT-4o-mini
- API REST (FastAPI)
- Bot Telegram integrado
- Logging de decisiones y metadata

---

## 🧱 Estructura del Proyecto

chatbot-ies/
├── app/
│ ├── main.py # API FastAPI
│ ├── retriever.py # Recuperación híbrida
│ ├── response_selector.py # Lógica de selección
│ ├── generator.py # Integración con LLM
│ ├── dialogue_manager.py # (Futuro) memoria de sesión
│ └── utils.py
├── scripts/
│ ├── build_index.py
│ ├── debug_embeddings.py
│ ├── test_chatbot.py
│ └── bot_telegram.py # Bot Telegram
├── data/
│ └── faqs.csv # Base de conocimiento
├── models/
│ ├── embeddings_index.faiss
│ └── faqs.pkl
├── logs/
│ └── chat_logs.json
├── requirements.txt
├── README.md
├── USERS_GUIDE.md
└── API_REFERENCE.md


---

## ✅ Instalación

--bash

git clone <repo>
cd chatbot-ies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## 🔧 Variables de Entorno

export GEN_BACKEND=ollama
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3
export TELEGRAM_BOT_TOKEN="tu_token"

## Correr el servidor HTTP

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


## 🤖 Correr el bot de Telegram

python3 -m scripts.bot_telegram


## 🧪 Test Manual del Selector

python3 -m scripts.test_chatbot

## 🎯 Versionado Actual

- v0.1 — API funcional
- v0.2 — Integración Telegram + selector generativo/clarify

## 🧭 Contribuir

- Crear PR con descripción de cambios
- Antes de merge:
  - Correr test_chatbot
  - Validar que el bot responda correctamente
  - Revisar logs

## 📄 Licencia

Proyecto académico para demostración interna del IES.