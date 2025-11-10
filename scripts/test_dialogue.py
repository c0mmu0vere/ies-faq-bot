import os
import sys
from pprint import pprint

# Ajustar path para importar desde /app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.dialogue_manager import handle_input, update_memory, get_recent_turns

user_id = "usuario_demo"

print("Simulación de diálogo con memoria y contexto")
print("Escribí un mensaje (o 'salir'):\n")

while True:
    user_msg = input("Usuario: ").strip()
    if user_msg.lower() in ['salir', 'exit', 'quit']:
        break

    # Simulamos respuesta (falsa) del bot para alimentar la memoria
    bot_msg = f"Bot respondió algo sobre: {user_msg}"

    # Actualizar memoria con el turno actual
    update_memory(user_id, user_msg, bot_msg)

    # Procesar entrada con posible referencia
    reformulada = handle_input(user_id, user_msg)

    print(f"\n🔁 Consulta reformulada o final: {reformulada}")
    print(f"🧠 Memoria actual (últimos 3 turnos):")
    pprint(get_recent_turns(user_id, n=3))
    print("-" * 60)