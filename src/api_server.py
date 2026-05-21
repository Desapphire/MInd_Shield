"""
Optional local-only FastAPI service for Mind-Shield+.
"""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI

from local_storage import LocalStorage


app = FastAPI(title="Mind-Shield+ Local API", docs_url=None, redoc_url=None)
storage = LocalStorage()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "local-ok", "privacy": "local-only"}


@app.post("/ingest")
def ingest(payload: Dict) -> Dict[str, str]:
    storage.save_payload("fatigue_logs", payload, ts=payload.get("ts"))
    return {"status": "stored"}


@app.get("/privacy")
def privacy() -> Dict:
    return {"local_only": True, "telemetry": False}
