import joblib

# Load model
model = joblib.load("model.pkl")

# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")

def predict_intent(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    return prediction[0]