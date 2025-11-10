#!/usr/bin/env python3
# scripts/bot_telegram.py

import os
import logging
import asyncio
from typing import Dict

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Integramos directamente con tus módulos
from app.retriever import encode_query, buscar_similares
from app.response_selector import seleccionar_respuesta, SelectorConfig

# ---------------- Logging ----------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("ies-bot")

# ---------------- Estado por chat ----------------
DEBUG_CHATS: Dict[int, bool] = {}  # chat_id -> debug_enabled


# --------------- Helpers ---------------
def _selector_cfg() -> SelectorConfig:
    # Usa los mismos umbrales que ya calibraste
    return SelectorConfig(
        tau_high=0.80,
        tau_low=0.55,
        near_tie_delta=0.05,
        show_k=3,
    )

async def _send_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int, seconds: float = 0.4):
    """Muestra 'typing...' un ratito (cosmético)."""
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(seconds)
    except Exception:
        pass


# --------------- Handlers ---------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    msg = (
        "¡Hola! Soy el Chatbot IES (v0.1)\n"
        "Enviame una consulta y te ayudo con respuestas basadas en las FAQs.\n\n"
        "Comandos:\n"
        "• /help – ver ayuda\n"
        "• /debug – alterna modo debug por chat (muestra metadatos)\n"
    )
    await update.message.reply_text(msg)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Ayuda – Chatbot IES\n\n"
        "• Escribí tu pregunta tal como la harías en el sitio (ej: “¿En qué ámbitos puede trabajar un Técnico Superior en RRHH?”).\n"
        "• El bot combina búsqueda semántica y léxica, y puede pedir aclaraciones si detecta empate.\n"
        "• /debug – alterna modo debug por chat para ver puntajes y modo del selector.\n"
    )
    await update.message.reply_text(msg)

async def debug_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    prev = DEBUG_CHATS.get(chat_id, False)
    DEBUG_CHATS[chat_id] = not prev
    state = "ON ✅" if DEBUG_CHATS[chat_id] else "OFF ❌"
    await update.message.reply_text(f"Modo debug: {state}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None or update.message.text is None:
        return

    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    if not text:
        await update.message.reply_text("Decime tu consulta y te doy una mano.")
        return

    # Muestra "typing..." mientras trabaja
    await _send_typing(context, chat_id, seconds=0.3)

    try:
        # 1) encode + recuperación híbrida (IMPORTANTE: pasar query_text para híbrido)
        qvec = encode_query(text)
        cands = buscar_similares(qvec, top_k=5, query_text=text)

        # 2) selección (extractive / generative / tie-break / fallback)
        cfg = _selector_cfg()
        sel = seleccionar_respuesta(
            query=text,
            candidatos=cands,
            cfg=cfg,
            enable_generation=True,  # usa GEN_BACKEND (ollama/openai/mock)
        )

        answer = sel.get("answer", "").strip()
        mode = sel.get("mode", "extractive")
        meta = sel.get("meta", {})

        # Render básico
        await update.message.reply_text(answer)

        # Meta opcional
        if DEBUG_CHATS.get(chat_id, False):
            # Mostramos un mini resumen con puntajes y top1
            best_dense = meta.get("best_dense")
            top1 = meta.get("top1_faq")
            backend = meta.get("generator_backend", "n/a")
            used_gen = meta.get("used_generator")
            dbg = (
                f"*Modo*: `{mode}`\n"
                f"*Top1*: {top1}\n"
                f"*best_dense*: `{best_dense}`\n"
                f"*gen_backend*: `{backend}`  *used_gen*: `{used_gen}`"
            )
            await update.message.reply_text(dbg, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        log.exception("Error procesando mensaje: %s", e)
        await update.message.reply_text(
            "Uy, algo falló al procesar tu mensaje. Probá de nuevo en un momento."
        )


# --------------- Main ---------------
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el entorno.")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("debug", debug_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Iniciando bot de Telegram (long polling)…")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()