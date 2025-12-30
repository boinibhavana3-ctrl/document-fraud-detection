import os
import cv2
import pytesseract
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------- FUNCTIONS ----------

def preprocess_image(image_path):
    """Load image, convert to gray, resize"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    return resized

def extract_text(image_path):
    """Extract text using OCR"""
    image = preprocess_image(image_path)
    return pytesseract.image_to_string(image)

def extract_image_features(image_path):
    """Flatten image as feature"""
    image = preprocess_image(image_path)
    return image.flatten()

def extract_text_features(text):
    """Simple text-based feature"""
    return len(text)

# ---------- DATASET LOADING ----------
def load_dataset(dataset_path="dataset"):
    X, y = [], []
    for label, folder in enumerate(["genuine", "fraudulent"]):
        folder_path = os.path.join(dataset_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            img_feat = extract_image_features(file_path)
            text = extract_text(file_path)
            text_feat = extract_text_features(text)

            features = np.append(img_feat, text_feat)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# ---------- TRAIN MODEL ----------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "fraud_model.pkl")
    print("Model saved as fraud_model.pkl")
    return model

# ---------- PREDICT ----------
def predict_document(model, image_path):
    img_feat = extract_image_features(image_path)
    text = extract_text(image_path)
    text_feat = extract_text_features(text)

    features = np.append(img_feat, text_feat).reshape(1, -1)
    result = model.predict(features)
    return "Genuine Document" if result[0]==0 else "Fraudulent Document"

# ---------- MAIN ----------
if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset("dataset")
    print("Dataset loaded. Training model...")
    model = train_model(X, y)

    while True:
        path = input("Enter document image path (or 'exit' to quit): ")
        if path.lower() == "exit":
            break
        if not os.path.exists(path):
            print("File not found! Try again.")
            continue
        result = predict_document(model, path)
        print("Prediction:", result)
