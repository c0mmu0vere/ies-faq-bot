# 🧠 IES FAQ Bot  
### Chatbot de preguntas frecuentes basado en NLP  

---

## 📌 Descripción

**IES FAQ Bot** es un chatbot desarrollado como proyecto académico para la materia *Procesamiento del Lenguaje Natural* de la Tecnicatura en Ciencia de Datos e Inteligencia Artificial (IES Siglo XXI).  
Su objetivo es brindar respuestas automáticas a las consultas frecuentes de estudiantes y aspirantes del IES, aplicando técnicas de **NLP** para identificar preguntas similares formuladas de manera diferente.

El bot utiliza un **motor de recuperación semántica** basado en embeddings de oraciones y la métrica de **similitud coseno** para determinar qué respuesta de la base de FAQs se asemeja más a la pregunta del usuario.  
Además, puede integrarse con **Telegram** para ofrecer una experiencia conversacional accesible y en tiempo real.

---

## 🎯 Objetivos del proyecto

- Construir un bot capaz de responder preguntas frecuentes sobre el IES de forma automática.  
- Aplicar modelos de *embeddings semánticos* (Sentence Transformers / Hugging Face).  
- Implementar una búsqueda semántica utilizando la métrica de similitud coseno.  
- Incorporar un umbral de confianza y un sistema de respuesta genérica (*fallback*).  
- Integrar el bot con Telegram para una interfaz de usuario sencilla.  
- (Opcional) Reescribir las respuestas de forma más natural con un modelo generativo.

---

## ⚙️ Tecnologías utilizadas

| Área | Herramienta / Librería |
|------|------------------------|
| Lenguaje principal | Python 3.10+ |
| Embeddings semánticos | SentenceTransformers (Hugging Face) |
| Similitud coseno | Numpy / Scikit-learn |
| Bot | python-telegram-bot |
| API / Webhook (opcional) | FastAPI |
| Despliegue | Render / Railway / Fly.io |
| Extras | Transformers (para reescritura generativa) |

---

## 🧩 Arquitectura del sistema

Usuario (Telegram)
↓
Bot (python-telegram-bot)
↓
Motor NLP (SentenceTransformers)
↓
Comparación semántica (Similitud Coseno + umbral)
↓
Respuesta (directa | generativa | fallback)

---

## 📂 Estructura del repositorio

ies-faq-bot/
├─ app/
│ ├─ main.py
│ ├─ nlp_core.py
│ ├─ bot_handlers.py
│ └─ config.py
├─ data/
│ ├─ faqs.csv
│ └─ embeddings.npz
├─ notebooks/
│ └─ 01_build_embeddings.ipynb
├─ tests/
│ └─ test_similarity.py
├─ REPORTE/
│ └─ (documentación futura)
├─ requirements.txt
└─ README.md

---

## 🚀 Instalación y ejecución

# 1. Clonar el repositorio
git clone https://github.com/<tu_usuario>/ies-faq-bot.git
cd ies-faq-bot

# 2. Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
