import joblib
import os

def predict_email(text):
    model = joblib.load("models/spam_classifier.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sample_prediction.txt", "w") as f:
        f.write(f"Email Text:\n{text}\n\n")
        f.write(f"Prediction: {prediction}\n")

    return prediction
