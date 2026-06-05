# 📧 Spam Classifier

A machine learning web app that detects whether a message is **spam or legitimate (ham)** — built with FastAPI, scikit-learn, and a clean glassmorphism UI.

---

## 🧠 How It Works

1. User pastes a message into the web UI
2. The text is cleaned (lowercased, URLs and punctuation removed)
3. It's vectorized using **TF-IDF**
4. A **Logistic Regression** model predicts: `SPAM` or `HAM`
5. Confidence score is returned and displayed

---

## 🗂️ Project Structure

```
spam-classifier/
├── app.py                # FastAPI backend
├── main.py               # Model training script
├── spam_model.pkl        # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Python dependencies
├── render.yaml           # Render deployment config
├── templates/
│   └── index.html        # Frontend HTML
└── static/
    ├── style.css         # Styles
    └── script.js         # Frontend logic
```

---

## 🛠️ Tech Stack

| Layer      | Technology                  |
|------------|-----------------------------|
| Backend    | FastAPI + Uvicorn           |
| ML Model   | scikit-learn (Logistic Regression + TF-IDF) |
| Frontend   | HTML, CSS, Vanilla JS       |
| Deployment | Render                      |

---

## 💻 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
cd spam-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the server**
```bash
uvicorn app:app --reload
```

**4. Open in browser**
```
http://127.0.0.1:8000
```

---

## 🔁 Retrain the Model (Optional)

If you want to retrain on your own data:

```bash
# Place your dataset at data/spam.csv (tab-separated: label, message)
python main.py
```

This will regenerate `spam_model.pkl` and `vectorizer.pkl`.

---

## 📡 API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "Congratulations! You won a free iPhone!"
}
```

**Response:**
```json
{
  "prediction": "SPAM",
  "confidence": 98.73
}
```

**Health check:** `GET /health` → `{ "status": "ok" }`

---

## 🧪 Test Messages

**Should return SPAM:**
- `Congratulations! You've won a FREE iPhone! Click here to claim now.`
- `URGENT: Your bank account has been suspended. Verify immediately.`
- `You have been selected for a $1000 gift card. Reply WIN to claim!`

**Should return HAM:**
- `Hey, are we still on for lunch tomorrow at 1pm?`
- `Can you send me the meeting notes from this morning?`
- `I'll be home by 7, save me some dinner!`

---

## 📄 License

MIT License — free to use and modify.
