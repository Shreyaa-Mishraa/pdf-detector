"""FastAPI web app: drag-and-drop static PDF triage.

  GET  /      serves static/index.html
  POST /scan  accepts one PDF, extracts static features, returns the verdict

Run locally:  uvicorn app:app --host 127.0.0.1 --port 8000

The upload is written to a random temp path (the client's filename is never
used), scanned statically, and deleted in a finally block even on errors.
Nothing in the PDF is executed or rendered.
"""

import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from features import FEATURE_NAMES, KEYWORDS, extract_features, to_vector

ROOT = Path(__file__).resolve().parent
# MODEL_DIR lets you point the app at another model.joblib (e.g. for testing).
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT))
MAX_BYTES = 10 * 1024 * 1024
CHUNK = 1024 * 1024
THRESHOLD = 0.5  # must match train.py

model = None


@asynccontextmanager
async def lifespan(_app):
    global model
    names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    if names != FEATURE_NAMES:
        raise RuntimeError("feature_names.json does not match features.py; retrain the model")
    model = joblib.load(MODEL_DIR / "model.joblib")
    yield


app = FastAPI(title="PDF malware triage", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    fd, tmp_path = tempfile.mkstemp(prefix="pdfscan_", suffix=".bin")
    try:
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(CHUNK):
                if size == 0 and not chunk.startswith(b"%PDF"):
                    raise HTTPException(400, "Not a PDF: file must start with %PDF")
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "File is larger than 10 MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty file")

        feats = extract_features(tmp_path)
        proba = float(model.predict_proba([to_vector(feats)])[0][1])
        return {
            "verdict": "malicious" if proba >= THRESHOLD else "benign",
            "probability": round(proba, 4),
            "keyword_counts": {k: feats[k] for k in KEYWORDS},
            "file_size": feats["file_size"],
            "parsed_ok": bool(feats["parsed_ok"]),
        }
    finally:
        await file.close()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
