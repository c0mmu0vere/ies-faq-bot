from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math
import json
from pathlib import Path
from datetime import datetime
from app.generator import get_backend_name
import re

_STOP_ES = {"de","la","el","los","las","y","o","u","en","del","al","para","por","con","un","una","que","es","son","se","a","lo"}

def _token_set(s: str):
    toks = re.findall(r"\w+", s.lower(), flags=re.UNICODE)
    return {t for t in toks if t not in _STOP_ES and len(t) > 1}

def jaccard(a: str, b: str) -> float:
    A = _token_set(a)
    B = _token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union

# Si querés desactivar la rama generativa mientras probamos:
try:
    from app.generator import rewrite_answer  # función opcional
    HAS_GENERATOR = True
except Exception:
    HAS_GENERATOR = False

LOG_PATH = Path("logs/chat_logs.json")


@dataclass(frozen=True)
class SelectorConfig:
    # Umbrales de decisión (ajustables)
    tau_high: float = 0.78   # “alto” → respuesta extractiva directa
    tau_low:  float = 0.55   # “medio” → reformulación generativa o desambiguación

    # Desambiguación (evita “falsos positivos” si hay empate cercano)
    near_tie_delta: float = 0.05  # si top1 - top2 < delta y ambos >= tau_low => pedir aclaración

    # Top-k a considerar para mostrar en fallback/desambiguación
    show_k: int = 3

    # Mensajes por defecto
    fallback_msg: str = "No estoy 100% seguro. ¿Podrías ser más específico?"
    tie_msg_prefix: str = "Encontré varias opciones parecidas. ¿Cuál de estas quisiste decir?"
    # Plantilla para opciones:
    tie_option_format: str = "- {i}. {pregunta}"


def _safe_score(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return -1.0
        return v
    except Exception:
        return -1.0


def _append_log(payload: Dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {**payload, "ts": datetime.utcnow().isoformat() + "Z"}
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _build_disambiguation_message(cands, cfg):
    """
    Fallback de tie-break SIN LLM:
    - Redacta una pregunta breve y natural
    - Muestra 2–3 opciones con bullets
    """
    show_k = getattr(cfg, "show_k", 3)
    opts_src = cands[:show_k]
    opts = "\n".join(
        f"- {c.get('pregunta_faq','')}"
        for c in opts_src
        if isinstance(c, dict) and c.get("pregunta_faq")
    )
    return (
        "¿Podrías aclarar tu consulta?\n\n"
        "Por favor, respóndeme con alguna de estas opciones:\n"
        f"{opts}"
    )

def _build_fallback_message(cands: List[Dict[str, Any]], cfg: SelectorConfig) -> str:
    sugerencias = []
    limit = min(len(cands), cfg.show_k)
    for i in range(limit):
        sugerencias.append(f"- {cands[i]['pregunta_faq']} (score={cands[i]['score']:.2f})")
    if sugerencias:
        return f"{cfg.fallback_msg}\nSugerencias:\n" + "\n".join(sugerencias)
    return cfg.fallback_msg


def seleccionar_respuesta(
    query: str,
    candidatos: List[Dict[str, Any]],
    cfg: Optional[SelectorConfig] = None,
    enable_generation: bool = True,
) -> Dict[str, Any]:
    """
    Decide el modo de respuesta en base a los scores de recuperación semántica (coseno).
    Respeta el ORDEN híbrido (RRF) que trae retriever y toma decisiones con coseno denso.
    """
    cfg = cfg or SelectorConfig()

    # Normalizamos/validamos scores
    cands = [
        {**c, "score": _safe_score(c.get("score", -1))}
        for c in candidatos if "respuesta" in c and "pregunta_faq" in c
    ]
    # 🚫 NO reordenar por score: ya vienen en orden híbrido (RRF)
    # cands = sorted(cands, key=lambda x: x["score"], reverse=True)

    # Top-k respetando el orden híbrido
    top_k = cands[: cfg.show_k]

    if not top_k:
        answer = _build_fallback_message(cands, cfg)
        meta = {"decision": "fallback", "reason": "no_candidates",
                "tau_low": cfg.tau_low, "tau_high": cfg.tau_high}
        _append_log({"query": query, "mode": "fallback", "meta": meta})
        return {"mode": "fallback", "answer": answer, "meta": meta}

    # Primer candidato según RRF
    top1 = top_k[0]
    best_dense = float(top1.get("score_dense", top1.get("score", 0.0)))
    best_fused = float(top1.get("score_fused", 0.0))

    # Segundo para evaluar empate cercano (coseno denso)
    second_dense = float(top_k[1].get("score_dense", top_k[1].get("score", 0.0))) if len(top_k) > 1 else -1.0

    # ============== Política de decisión ==============

    # 1) Extractivo (alto)
    if best_dense >= cfg.tau_high:
        answer = top1["respuesta"]
        meta = {
            "decision": "extractive",
            "best_dense": best_dense,
            "second_dense": second_dense,
            "delta_dense": (best_dense - second_dense) if second_dense >= 0 else None,
            "top1_faq": top1["pregunta_faq"],
            "top1_fused": best_fused,
            "tau_low": cfg.tau_low,
            "tau_high": cfg.tau_high,
            "ranking": top_k,
        }
        _append_log({"query": query, "mode": "extractive", "meta": meta})
        return {"mode": "extractive", "answer": answer, "meta": meta}

    # 2) Tie-break (empate cercano) — SOLO CLARIFY
    if (
        best_dense >= cfg.tau_low
        and second_dense >= cfg.tau_low
        and (best_dense - second_dense) < cfg.near_tie_delta
    ):
        used_gen = False
        if enable_generation and HAS_GENERATOR:
            contexto = [{"pregunta_faq": c["pregunta_faq"], "respuesta": c["respuesta"]} for c in top_k]
            try:
                clarify_ctx = json.dumps(contexto, ensure_ascii=False)
                gen_answer = rewrite_answer(query=query, base_answer="", context=clarify_ctx, mode="clarify")
                answer = gen_answer.strip() if gen_answer and gen_answer.strip() else _build_disambiguation_message(top_k, cfg)
                used_gen = bool(gen_answer and gen_answer.strip())
            except Exception:
                answer = _build_disambiguation_message(top_k, cfg)
        else:
            answer = _build_disambiguation_message(top_k, cfg)

        meta = {
            "decision": "tie-break",
            "best_dense": best_dense,
            "second_dense": second_dense,
            "delta_dense": (best_dense - second_dense) if second_dense >= 0 else None,
            "top1_faq": top1["pregunta_faq"],
            "top1_fused": best_fused,
            "tau_low": cfg.tau_low,
            "tau_high": cfg.tau_high,
            "used_generator": used_gen,
            "ranking": top_k,
            "generator_backend": get_backend_name() if used_gen else None
        }
        _append_log({"query": query, "mode": "tie-break", "meta": meta})
        return {"mode": "tie-break", "answer": answer, "meta": meta}

    # 3) Generative (polish) — zona intermedia con gate léxico
    if cfg.tau_low <= best_dense < cfg.tau_high:
        # Gate léxico para evitar polish cuando la consulta y el top1 difieren demasiado
        lex_overlap = jaccard(query, top1["pregunta_faq"])
        MIN_JACCARD = 0.22  # ajustar 0.20–0.25 si hace falta

        if lex_overlap < MIN_JACCARD:
            # Preferimos aclaración (tie-break) antes que pulir algo potencialmente distinto
            show_k = getattr(cfg, "show_k", 3)
            opts_src = top_k[:show_k]
            contexto = [
                {"pregunta_faq": c["pregunta_faq"], "respuesta": c["respuesta"]}
                for c in opts_src
            ]

            used_gen = False
            clarify_text = None
            if enable_generation and HAS_GENERATOR:
                try:
                    ctx_str = json.dumps(contexto, ensure_ascii=False)
                    clarify_text = rewrite_answer(
                        query=query,
                        base_answer="",
                        context=ctx_str,
                        mode="clarify",
                    )
                    used_gen = True if (clarify_text and clarify_text.strip()) else False
                except Exception:
                    clarify_text = None
                    used_gen = False

            if not clarify_text:
                opts = "\n".join(f"- {c['pregunta_faq']}" for c in opts_src)
                clarify_text = (
                    "¿Podrías aclarar tu consulta?\n\n"
                    "Por favor, respóndeme con alguna de estas opciones:\n"
                    f"{opts}"
                )

            meta = {
                "decision": "tie-break",
                "reason": "low_lexical_overlap_for_polish",
                "best_dense": best_dense,
                "second_dense": second_dense,
                "delta_dense": best_dense - second_dense,
                "lex_jaccard": lex_overlap,
                "tau_low": cfg.tau_low,
                "tau_high": cfg.tau_high,
                "near_tie_delta": getattr(cfg, "near_tie_delta", None),
                "used_generator": used_gen,
                "generator_backend": get_backend_name() if used_gen else None,
                "top1_faq": top1["pregunta_faq"],
                "top1_fused": best_fused,
                "ranking": top_k,
            }
            _append_log({"query": query, "mode": "tie-break", "meta": meta})
            return {"mode": "tie-break", "answer": clarify_text, "meta": meta}

        # Si pasa el gate, hacemos polish normal (prosa natural)
        used_gen = False
        answer = top1["respuesta"]  # si el LLM falla, devolvemos esto
        if enable_generation and HAS_GENERATOR:
            try:
                ctx_str = json.dumps(
                    [{"pregunta_faq": c["pregunta_faq"], "respuesta": c["respuesta"]} for c in top_k],
                    ensure_ascii=False
                )
                polished = rewrite_answer(
                    query=query,
                    base_answer=top1["respuesta"],
                    context=ctx_str,
                    mode="polish",
                )
                if polished and polished.strip():
                    answer = polished.strip()
                    used_gen = True
            except Exception:
                pass

        meta = {
            "decision": "generative",
            "best_dense": best_dense,
            "second_dense": second_dense,
            "delta_dense": best_dense - second_dense,
            "lex_jaccard": lex_overlap,
            "top1_faq": top1["pregunta_faq"],
            "top1_fused": best_fused,
            "tau_low": cfg.tau_low,
            "tau_high": cfg.tau_high,
            "near_tie_delta": getattr(cfg, "near_tie_delta", None),
            "used_generator": used_gen,
            "generator_backend": get_backend_name() if used_gen else None,
            "ranking": top_k,
        }
        _append_log({"query": query, "mode": "generative", "meta": meta})
        return {"mode": "generative", "answer": answer, "meta": meta}


    # 3b) Generative borderline (laboral cercano a tau_low)
    q_lower = (query or "").lower()
    laboral_hint = any(k in q_lower for k in [
        "salida", "salidas", "laboral", "trabajo", "trabajar",
        "áreas", "ambitos", "ámbitos", "egresad", "oportunidades",
        "campos", "campo laboral", "puestos", "empleo", "empleabilidad",
        "recursos humanos", "rrhh"
    ])
    if laboral_hint and (cfg.tau_low - 0.05) <= best_dense < cfg.tau_low:
        used_gen = False
        answer = top1["respuesta"]
        if enable_generation and HAS_GENERATOR:
            contexto = [{"pregunta_faq": c["pregunta_faq"], "respuesta": c["respuesta"]} for c in top_k]
            try:
                ctx_str = json.dumps(contexto, ensure_ascii=False)
                polished = rewrite_answer(query=query, base_answer=top1["respuesta"], context=ctx_str, mode="polish")
                if polished and polished.strip():
                    answer = polished.strip()
                    used_gen = True
            except Exception:
                pass

        meta = {
            "decision": "generative-borderline",
            "best_dense": best_dense,
            "top1_faq": top1["pregunta_faq"],
            "tau_low": cfg.tau_low,
            "tau_high": cfg.tau_high,
            "used_generator": used_gen,
            "ranking": top_k,
            "generator_backend": get_backend_name() if used_gen else None
        }
        _append_log({"query": query, "mode": "generative", "meta": meta})
        return {"mode": "generative", "answer": answer, "meta": meta}

    # 4) Fallback (orden híbrido)
    answer = _build_fallback_message(top_k, cfg)
    meta = {
        "decision": "fallback",
        "best_dense": best_dense,
        "second_dense": second_dense,
        "top1_faq": top1["pregunta_faq"],
        "top1_fused": best_fused,
        "tau_low": cfg.tau_low,
        "tau_high": cfg.tau_high,
        "ranking": top_k,
    }
    _append_log({"query": query, "mode": "fallback", "meta": meta})
    return {"mode": "fallback", "answer": answer, "meta": meta}