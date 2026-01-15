from sklearn.metrics import accuracy_score, classification_report
import os

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("Evaluation saved to outputs/evaluation_report.txt")
