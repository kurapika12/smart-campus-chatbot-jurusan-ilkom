import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_excel("dataset_pertanyaan_chatbot.xlsx")

# Ambil data
X = df["question"]
y = df["intent"]

# TF-IDF
vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)

# Model
model = LogisticRegression()

# Training
model.fit(X_vector, y)

# Simpan model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model berhasil di-training!")