# Guía de Uso — Chatbot IES

## 👋 Bienvenido/a

Este chatbot está diseñado para ayudarte con preguntas sobre las carreras y servicios del IES. Funciona tanto en Telegram como a través de la API web.

---

## 🧠 ¿Qué puede responder?

- Información sobre carreras (duración, perfil, título, materias)
- Modalidades (presencial, distancia)
- Recursos (laboratorios, biblioteca)
- Trámites del alumno
- Información de prácticas y pasantías

---

## 💬 ¿Cómo preguntar?

Escribí tu pregunta como lo harías normalmente:

- “¿En qué ámbitos puede trabajar un Técnico Superior en RRHH?”
- “¿Cuánto dura la carrera de Recursos Humanos?”
- “Modalidades de cursado de Videojuegos”
- “Requisitos para pasantías”

---

## 🤖 Modos de respuesta del bot

1. **Extractive**  
   La consulta coincide perfectamente y el bot devuelve la respuesta exacta.

2. **Polish (generative)**  
   La consulta es equivalente a una FAQ pero con otra redacción.  
   El bot devuelve una reformulación suave y coherente.

3. **Clarify (tie-break)**  
   Hay varias respuestas candidatas.  
   El bot te pedirá aclarar cuál quisiste preguntar y te mostrará opciones.

4. **Fallback**  
   El bot no encontró suficiente información.  
   Te sugerirá preguntas relacionadas.

---

## ⚠️ ¿Qué evitar?

- Preguntas muy vagas (“información”, “carreras”)
- Preguntas que mezclan varios temas
- Errores ortográficos graves
- Consultas que no estén en ninguna FAQ

---

## 🧪 Modo debug

En Telegram:

/debug

Esto muestra puntajes, el modo seleccionado y metadatos útiles.

---

## ✅ Consejos para mejores resultados

- Nombrar la carrera exactamente
- Usar frases simples
- Evitar siglas desconocidas

---

## 🆘 ¿Necesitás ayuda?

En Telegram:

/help