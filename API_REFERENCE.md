# API Reference — Chatbot IES

## Base URL

http://127.0.0.1:8000

---

# 📡 Endpoints

---

## ✅ GET `/health`

Verifica el estado del servidor.

### Respuesta

{
  "status": "ok",
  "version": "0.1"
}

## ✅ POST /chat

Solicitud

Envía una consulta del usuario y devuelve la respuesta del chatbot.

Request Body

{
  "query": "¿En qué ámbitos puede trabajar un Técnico Superior en Recursos Humanos?",
  "session_id": "opcional",
  "top_k": 5,
  "enable_generation": true
}

Respuesta

{
  "mode": "extractive",
  "answer": "Podés desempeñarte en áreas de gestión de personas...",
  "meta": {
    "decision": "extractive",
    "best_dense": 1.0,
    "second_dense": 0.88,
    "delta_dense": 0.11,
    "top1_faq": "¿En qué ámbitos puede trabajar un Técnico Superior en Recursos Humanos?",
    "ranking": [
      {
        "faq_id": "378",
        "pregunta_faq": "¿En qué ámbitos puede trabajar un Técnico Superior en Recursos Humanos?",
        "score_dense": 1.0,
        "score_lex": 0.52,
        "score_fused": 0.18
      }
    ],
    "generator_backend": "ollama",
    "used_generator": true
  }
}

## 🔁 Reconstruir índice FAISS

Cada vez que modifiques data/faqs.csv:

python3 -m scripts.build_index

## 🤖 Integración con frontend

Ejemplo en JavaScript:

const res = await fetch("http://127.0.0.1:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "modalidad de videojuegos" })
});
const data = await res.json();
console.log(data.answer);

## ✅ Errores comunes

- used_generator: false → LLM no configurado
- 500 → Ollama no está corriendo
- 400 → Campo query vacío

## 🔒 Seguridad
- Limitar CORS si se despliega en producción
- No exponer el TOKEN de Telegram
- Añadir rate limiting

## 📄 Versión

- API versión: v0.2