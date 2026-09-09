"""FastAPI backend for the Emergent.ai frontend.

    uvicorn src.api:app --reload --port 8000

Endpoints
  GET  /health   liveness + which backends are actually loaded
  GET  /intents  the taxonomy, so the frontend can render labels without hardcoding
  POST /predict  the full evidence-backed support decision
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import llm
from intents import taxonomy_table
from agent import get_agent

app = FastAPI(title=f"{C.BRAND} AI Support Agent", version="1.0.0")
# Open CORS: this is a local demo backend for a hosted frontend builder, with no
# auth and no user data. Lock this down before anything resembling production.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         json_schema_extra={"example": "@SpotifyCares you charged me twice this month"})
    top_k: int = Field(C.RETRIEVAL_K, ge=1, le=10)


class EvidenceItem(BaseModel):
    customer_message: str
    historical_reply: str
    similarity: float
    historical_intent: str
    is_boilerplate: bool


class PredictResponse(BaseModel):
    intent: str
    intent_confidence: float
    intent_runner_up: str
    intent_margin: float
    risk_level: str
    evidence_quality: float
    decision: str
    decision_reason: str
    reply: str
    reply_is_suggestion_only: bool
    evidence: list[EvidenceItem]
    evidence_signals: dict
    evidence_flags: list[str]
    critique: dict
    generation_backend: str
    calibrated: bool
    latency_ms: float
    trace: list[dict]


FRONTEND = C.ROOT / "frontend" / "index.html"
DEMO_CASES = C.ROOT / "demo_cases.json"


@app.get("/", include_in_schema=False)
def index():
    """Serve the single-page frontend from the same origin as the API.

    Same-origin means no CORS dance, no build step, no second server to start
    during a demo — `uvicorn src.api:app` is the whole stack.
    """
    if not FRONTEND.exists():
        return JSONResponse({"error": "frontend/index.html missing"}, status_code=404)
    return FileResponse(FRONTEND, media_type="text/html")


@app.get("/demo_cases")
def demo_cases():
    """Real messages from the held-out test split, for the frontend's demo chips.

    These are inputs only. Nothing about the expected output is stored here — the
    frontend always renders whatever the live agent returns.
    """
    if not DEMO_CASES.exists():
        return {"cases": []}
    return json.loads(DEMO_CASES.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    try:
        a = get_agent()
        ok = True
        n_cases = a.index.matrix.shape[0]
        thresholds = a.thresholds
        calibrated = a.calibrator is not None
    except Exception as e:
        return {"status": "degraded", "error": str(e),
                "hint": "run: python -m src.data_processing && python -m src.train "
                        "&& python -m src.retrieval && python -m src.calibration"}
    return {"status": "ok", "brand": C.BRAND, "model_loaded": ok,
            "historical_cases_indexed": n_cases, "calibrated": calibrated,
            "llm_backend": llm.backend_name(), "thresholds": thresholds}


@app.get("/intents")
def intents():
    return {"brand": C.BRAND, "intents": taxonomy_table()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        st = get_agent().handle(req.message, k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent failure: {e}")
    return PredictResponse(
        intent=st["intent"],
        intent_confidence=st["intent_confidence"],
        intent_runner_up=st["runner_up"],
        intent_margin=st["margin"],
        risk_level=st["risk_level"],
        evidence_quality=st["evidence_quality"],
        decision=st["decision"],
        decision_reason=st["decision_reason"],
        reply=st["reply"],
        reply_is_suggestion_only=st["reply_is_suggestion_only"],
        evidence=[EvidenceItem(**{k: c[k] for k in
                                  ("customer_message", "historical_reply", "similarity",
                                   "historical_intent", "is_boilerplate")})
                  for c in st["historical_evidence"]],
        evidence_signals=st["evidence_signals"],
        evidence_flags=st["evidence_flags"],
        critique=st["critique"],
        generation_backend=st["backend"],
        calibrated=st["calibrated"],
        latency_ms=st["latency_ms"],
        trace=st["trace"],
    )
