import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...\n")

df = pd.read_csv(
    "data/spam.csv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["label"].value_counts())


# ==========================================
# CONVERT LABELS
# ham = 0
# spam = 1
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nLabels converted successfully!\n")


# ==========================================
# TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    # convert to lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("Cleaning messages...\n")

df["message"] = df["message"].apply(clean_text)

print("Text cleaning completed!\n")


# ==========================================
# FEATURES & LABELS
# ==========================================

X = df["message"]

y = df["label"]


# ==========================================
# SPLIT DATASET
# ==========================================

print("Splitting dataset into train and test...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset split completed!\n")


# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

print("Converting text into vectors using TF-IDF...\n")

vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)

print("Vectorization completed!\n")


# ==========================================
# TRAIN MODEL
# ==========================================

print("Training Logistic Regression model...\n")

model = LogisticRegression()

model.fit(
    X_train_vectorized,
    y_train
)

print("Model training completed!\n")


# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("Making predictions...\n")

predictions = model.predict(X_test_vectorized)


# ==========================================
# EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(f"\nAccuracy: {round(accuracy * 100, 2)}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================================
# SAVE MODEL & VECTORIZER
# ==========================================

print("Saving trained model...\n")

joblib.dump(
    model,
    "spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("Model and vectorizer saved successfully!\n")


# ==========================================
# CUSTOM MESSAGE TESTING
# ==========================================

print("==========================================")
print("LIVE SPAM DETECTOR")
print("==========================================")

while True:

    user_input = input(
        "\nEnter a message to classify (or type 'quit' to exit): "
    )

    if user_input.lower() == "quit":
        print("\nExiting Spam Classifier...")
        break

    cleaned_input = clean_text(user_input)

    vector_input = vectorizer.transform(
        [cleaned_input]
    )

    prediction = model.predict(
        vector_input
    )[0]

    probability = model.predict_proba(
        vector_input
    )[0]

    spam_confidence = round(
        probability[1] * 100,
        2
    )

    good_confidence = round(
        probability[0] * 100,
        2
    )

    print("\n==========================================")

    if prediction == 1:
        print("Prediction: SPAM")
        print(f"Spam Confidence: {spam_confidence}%")
    else:
        print("Prediction: GOOD")
        print(f"ood Confidence: {good_confidence}%")

    print("==========================================")