# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Integraciones internas
from app.retriever import encode_query, buscar_similares
from app.response_selector import seleccionar_respuesta, SelectorConfig

# -------- FastAPI setup --------
app = FastAPI(title="IES FAQ Chatbot API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajustar si querés restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5
    enable_generation: Optional[bool] = True

class ChatResponse(BaseModel):
    mode: str
    answer: str
    meta: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1) encode + recuperar (IMPORTANTE: pasar query_text para híbrido)
    qvec = encode_query(req.query)
    cands = buscar_similares(qvec, top_k=req.top_k or 5, query_text=req.query)

    # 2) seleccionar (selector ya maneja extractive/generative/tie-break/fallback)
    cfg = SelectorConfig(
        tau_high=0.80,
        tau_low=0.55,          # tu ajuste actual
        near_tie_delta=0.05,
        show_k=3
    )

    sel = seleccionar_respuesta(
        query=req.query,
        candidatos=cands,
        cfg=cfg,
        enable_generation=bool(req.enable_generation),
    )

    return ChatResponse(mode=sel["mode"], answer=sel["answer"], meta=sel["meta"])
