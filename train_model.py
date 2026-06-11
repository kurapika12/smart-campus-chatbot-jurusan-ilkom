import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ======================
# LOAD DATASET
# ======================

df = pd.read_excel("dataset_pertanyaan_chatbot_v2.xlsx")

print("\n===== INFORMASI DATASET =====")

texts = df["question"].astype(str).str.lower()
labels = df["intent"].astype(str)

print(f"Jumlah Data   : {len(df)}")
print(f"Jumlah Intent : {labels.nunique()}")

print("\nDistribusi Intent:")
print(labels.value_counts())

# ======================
# SPLIT DATA
# ======================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ======================
# TF-IDF
# ======================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3),
    min_df=1,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ======================
# TRAIN MODEL
# ======================

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ======================
# PREDIKSI
# ======================

y_pred = model.predict(X_test)

# ======================
# EVALUASI
# ======================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\n" + "=" * 60)
print("HASIL EVALUASI MODEL")
print("=" * 60)

print(f"Data Training : {len(X_train_text)}")
print(f"Data Testing  : {len(X_test_text)}")

print("\nMETRIK EVALUASI")
print("-" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nCLASSIFICATION REPORT")
print("-" * 60)
print(report)

# ======================
# SIMPAN EVALUASI
# ======================

with open(
    "hasil_evaluasi.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("HASIL EVALUASI MODEL\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 60 + "\n")
    f.write(report)

# ======================
# SIMPAN MODEL
# ======================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel berhasil ditraining!")
print("Model disimpan ke model.pkl")
print("Vectorizer disimpan ke vectorizer.pkl")
print("Evaluasi disimpan ke hasil_evaluasi.txt")