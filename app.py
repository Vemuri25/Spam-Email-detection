from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import joblib
import re
import os

app = FastAPI()

# ── Resolve paths relative to this file ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR    = os.path.join(BASE_DIR, "static")
MODEL_PATH    = os.path.join(BASE_DIR, "spam_model.pkl")
VEC_PATH      = os.path.join(BASE_DIR, "vectorizer.pkl")

# ── Sanity-check on startup ───────────────────────────────────────────────────
for path, label in [
    (TEMPLATES_DIR, "templates/"),
    (STATIC_DIR,    "static/"),
    (os.path.join(TEMPLATES_DIR, "index.html"), "templates/index.html"),
    (MODEL_PATH,    "spam_model.pkl"),
    (VEC_PATH,      "vectorizer.pkl"),
]:
    if not os.path.exists(path):
        raise RuntimeError(f"MISSING: {label}  →  {path}")

# ── Load ML assets ────────────────────────────────────────────────────────────
model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# ── Mount static files & templates ───────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Message(BaseModel):
    text: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(message: Message):
    cleaned    = clean_text(message.text)
    vector     = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    confidence = round(float(probability[prediction]) * 100, 2)
    result     = "SPAM" if prediction == 1 else "HAM"
    return JSONResponse({"prediction": result, "confidence": confidence})


# ── Health check (useful for Render) ─────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}
