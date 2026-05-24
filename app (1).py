from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import joblib
import re
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR    = os.path.join(BASE_DIR, "static")
MODEL_PATH    = os.path.join(BASE_DIR, "spam_model.pkl")
VEC_PATH      = os.path.join(BASE_DIR, "vectorizer.pkl")

# ── Load ML assets with clear error if missing ────────────────────────────────
try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ FAILED to load model: {e}")
    raise RuntimeError(
        f"Could not load model files.\n"
        f"MODEL_PATH={MODEL_PATH}\n"
        f"VEC_PATH={VEC_PATH}\n"
        f"Error: {e}"
    )

# ── Mount static & templates ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Message(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(message: Message):
    cleaned     = clean_text(message.text)
    vector      = vectorizer.transform([cleaned])
    prediction  = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    confidence  = round(float(probability[prediction]) * 100, 2)
    result      = "SPAM" if prediction == 1 else "HAM"
    return JSONResponse({"prediction": result, "confidence": confidence})


@app.get("/health")
async def health():
    return {"status": "ok"}
