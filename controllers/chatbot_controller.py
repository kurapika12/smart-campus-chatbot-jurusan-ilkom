import pandas as pd

from models.chatbot_model import predict_intent

# Load response dataset
response_df = pd.read_excel(
    "dataset_response_chatbot.xlsx"
)

def get_chatbot_response(user_input):

    # Prediksi intent
    intent = predict_intent(user_input)

    # Cari response
    response_row = response_df[
        response_df["intent"] == intent
    ]

    # Jika ditemukan
    if not response_row.empty:

        return response_row.iloc[0]["response"]

    return "Maaf, jawaban belum tersedia."