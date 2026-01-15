from sklearn.naive_bayes import MultinomialNB
import joblib
import os

def train(X_train, y_train, vectorizer):
    model = MultinomialNB()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/spam_classifier.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    return model
