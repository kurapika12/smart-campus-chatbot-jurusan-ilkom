import pickle
import pandas as pd
import re

# ==========================
# LOAD MODEL
# ==========================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ==========================
# LOAD DATASET
# ==========================

response_df = pd.read_excel(
    "dataset_response_chatbot.xlsx"
)

jadwal_df = pd.read_excel(
    "jadwal_kuliah.xlsx"
)

ruangan_df = pd.read_excel(
    "ruangan_kosong.xlsx"
)

# ==========================
# CHATBOT
# ==========================

def get_chatbot_response(user_input):

    text = user_input.lower().strip()

    vector = vectorizer.transform([text])

    probs = model.predict_proba(vector)[0]

    confidence = probs.max()

    intent = model.classes_[probs.argmax()]

    print(
        f"Intent: {intent} | Confidence: {confidence:.2f}"
    )

    # Confidence threshold
    if confidence < 0.40:
        return (
            "Maaf, saya kurang memahami pertanyaan tersebut. "
            "Silakan gunakan kalimat yang lebih jelas."
        )

    # ======================
    # JADWAL KULIAH
    # ======================

    if intent == "jadwal_kuliah":

        angkatan_match = re.search(
            r"\b(23|24)\b",
            text
        )

        hari_match = re.search(
            r"senin|selasa|rabu|kamis|sabtu",
            text
        )

        if not angkatan_match:
            return "Silakan sebutkan angkatan."

        if not hari_match:
            return "Silakan sebutkan hari."

        angkatan = int(
            angkatan_match.group()
        )

        hari = hari_match.group()

        hasil = jadwal_df[
            (jadwal_df["angkatan"] == angkatan)
            &
            (jadwal_df["hari"] == hari)
        ]

        if hasil.empty:
            return "Jadwal tidak ditemukan."

        return hasil.iloc[0]["response"]

    # ======================
    # RUANGAN KOSONG
    # ======================

    elif intent == "ruangan_kosong":

        hari_match = re.search(
            r"senin|selasa|rabu|kamis|sabtu",
            text
        )

        if not hari_match:
            return (
                "Silakan sebutkan hari yang ingin dicek."
            )

        hari = hari_match.group()

        hasil = ruangan_df[
            ruangan_df["hari"] == hari
        ]

        if hasil.empty:
            return (
                "Data ruangan kosong tidak ditemukan."
            )

        return hasil.iloc[0]["response"]

    # ======================
    # INTENT LAIN
    # ======================

    hasil = response_df[
        response_df["intent"] == intent
    ]

    if hasil.empty:
        return (
            "Maaf, jawaban tidak ditemukan."
        )

    return hasil.iloc[0]["response"]