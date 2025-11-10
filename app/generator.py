import os
import json
from typing import List, Dict

SYSTEM_RULES = """
Eres un asistente de FAQ institucional. Debes:
- Responder solo con la información del contexto proporcionado (pares Q/A recuperados).
- No inventes datos ni agregues disclaimers del tipo "No dispongo de esa información".
- Si falta información, omite la parte faltante sin mencionarlo explícitamente.
- No mezcles tareas: si el modo es "clarify", solo pregunta y sugiere opciones; si es "polish", solo reescribe la respuesta base.
- No uses encabezados ni prefijos como "Modo polish" o "Modo clarify".
- Formato sobrio: frase(s) o viñetas simples; evita prosa redundante.
"""

def _ctx_to_qa_text(context_pairs: List[Dict], k: int = 3) -> str:
    return "\n\n".join(
        f"Q: {c.get('pregunta_faq','')}\nA: {c.get('respuesta','')}"
        for c in context_pairs[:k]
        if isinstance(c, dict)
    ).strip()

def build_prompt(query: str, base_answer: str, context_pairs: List[Dict], mode: str) -> str:
    ctx_str = _ctx_to_qa_text(context_pairs, k=3)

    if mode == "clarify":
        # Formato estrictamente controlado: una pregunta + 2-3 opciones, nada más.
        return (
            f"{SYSTEM_RULES}\n\n"
            f"Consulta del usuario:\n{query}\n\n"
            f"Contexto (pares Q/A):\n{ctx_str}\n\n"
            "Tarea (CLARIFY):\n"
            "1) Formula UNA sola pregunta breve para desambiguar.\n"
            "2) Luego ofrece 2-3 opciones cortas derivadas de los TÍTULOS del contexto.\n"
            "3) Devuelve SOLO este formato exacto, sin encabezados ni notas meta:\n"
            "Pregunta: <texto de una sola línea>\n"
            "Opciones:\n"
            "- <opción 1>\n"
            "- <opción 2>\n"
            "- <opción 3>\n"
        )

    if mode == "polish":
        # Reescritura concisa de la respuesta base; se permite 1-2 viñetas si aportan claridad.
        return (
            f"{SYSTEM_RULES}\n\n"
            f"Consulta del usuario:\n{query}\n\n"
            f"Contexto (pares Q/A):\n{ctx_str}\n\n"
            "Tarea (POLISH):\n"
            "Reescribe de forma clara y concisa la 'Respuesta base'. "
            "Puedes integrar 1–2 viñetas breves si ayudan a la claridad. "
            "No inventes datos. Devuelve SOLO el texto reescrito, sin encabezados ni notas meta.\n\n"
            f"Respuesta base:\n{base_answer}\n"
        )

    # Fallback: tratar como polish mínimo
    return (
        f"{SYSTEM_RULES}\n\n"
        f"Consulta del usuario:\n{query}\n\n"
        f"Contexto (pares Q/A):\n{ctx_str}\n\n"
        "Tarea: Reescribe la 'Respuesta base' de forma clara y breve.\n\n"
        f"Respuesta base:\n{base_answer}\n"
    )

def parse_context(context: str) -> List[Dict]:
    try:
        obj = json.loads(context)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return []

# ================== Backends ==================

class GeneratorBackend:
    def rewrite(self, query: str, base_answer: str, context_pairs: List[Dict], mode: str = "polish") -> str:
        raise NotImplementedError

class MockBackend(GeneratorBackend):
    def rewrite(self, query, base_answer, context_pairs, mode="polish") -> str:
        if mode == "clarify":
            opts = [c.get("pregunta_faq") for c in context_pairs[:3] if "pregunta_faq" in c]
            opt_str = "".join([f"\n- {o}" for o in opts]) if opts else ""
            return f"Pregunta: ¿Podrías aclarar tu consulta?\nOpciones:{opt_str}"
        return base_answer

# ---- OLLAMA (local) ----
class OllamaBackend(GeneratorBackend):
    def __init__(self, model: str = None, host: str = None, temperature: float = 0.2, max_tokens: int = 320):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")  # ya lo tenés descargado
        self.host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", temperature))
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", max_tokens))

    def rewrite(self, query, base_answer, context_pairs, mode="polish") -> str:
        import requests
        prompt = build_prompt(query, base_answer, context_pairs, mode)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": 2048
            },
            "stream": False
        }
        url = f"{self.host}/api/chat"
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict):
            msg = data.get("message", {})
            content = (msg or {}).get("content", "") or ""
            content = content.strip()
            if content:
                return content

        return base_answer or "Pregunta: ¿Podrías aclarar tu consulta?\nOpciones:\n- Opción A\n- Opción B"

# ---- OPENAI (hosted) ----
class OpenAIBackend(GeneratorBackend):
    def __init__(self, model: str = None, temperature: float = 0.2, max_tokens: int = 320):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", temperature))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", max_tokens))
        self.api_key = os.getenv("OPENAI_API_KEY")

    def rewrite(self, query, base_answer, context_pairs, mode="polish") -> str:
        if not self.api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en el entorno.")
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)

        prompt = build_prompt(query, base_answer, context_pairs, mode)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        text = resp.choices[0].message.content or ""
        return text.strip() if text else base_answer

# ====== Factory y API estable ======

def get_backend() -> GeneratorBackend:
    backend = os.getenv("GEN_BACKEND", "").lower().strip()
    if backend == "ollama":
        return OllamaBackend()
    if backend == "openai":
        return OpenAIBackend()
    return MockBackend()

def rewrite_answer(query: str, base_answer: str, context: str = "", mode: str = "polish") -> str:
    backend = get_backend()
    context_pairs = parse_context(context)
    return backend.rewrite(query=query, base_answer=base_answer, context_pairs=context_pairs, mode=mode)