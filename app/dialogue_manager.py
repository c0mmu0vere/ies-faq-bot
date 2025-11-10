# app/dialogue_manager.py
from collections import defaultdict, deque
import re

# Memoria de usuario: {user_id: deque([(user_msg, bot_msg), ...])}
user_memory = defaultdict(lambda: deque(maxlen=4))

def update_memory(user_id, user_msg, bot_msg):
    """
    Guarda el último turno de conversación (entrada del usuario y respuesta del bot)
    """
    user_memory[user_id].append((user_msg, bot_msg))


def get_recent_turns(user_id, n=3):
    """
    Devuelve los últimos `n` turnos (pares user-bot) del usuario.
    """
    return list(user_memory[user_id])[-n:]


def detect_reference(user_msg):
    """
    Detecta si el usuario hace una referencia implícita a una respuesta anterior.
    Ejemplos: "eso", "lo anterior", "repetí", "cuáles eran"
    """
    patrones = [
        r"\beso\b",
        r"\blo anterior\b",
        r"\brepet[íi]\b",
        r"\bcu[aá]les eran\b",
        r"\bme lo pod[eé]s repetir\b",
        r"\bla info\b",
        r"\bese dato\b"
    ]
    texto = user_msg.lower()
    return any(re.search(pat, texto) for pat in patrones)


def reformulate_query(user_msg, memory):
    """
    Si hay referencias implícitas, reformula la pregunta incluyendo el contexto anterior.
    """
    if not memory:
        return user_msg  # Sin memoria previa, no se puede reformular

    contexto = " ".join([turno[0] for turno in memory])  # concatena últimos mensajes del usuario
    return f"En base a lo anterior: {contexto}. Mi pregunta es: {user_msg}"


def handle_input(user_id, user_msg):
    """
    Flujo principal del Gestor de Diálogo:
    - Recupera historial
    - Detecta si debe reformular
    - Devuelve query final (explícita o no)
    """
    memoria = get_recent_turns(user_id)

    if detect_reference(user_msg):
        query_reformulada = reformulate_query(user_msg, memoria)
        return query_reformulada, True
    else:
        return user_msg, False