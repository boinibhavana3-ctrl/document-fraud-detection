import os

project_name = "Document-Fraud-Detection"

folders = [
    "dataset/genuine",
    "dataset/fraudulent",
    "preprocessing",
    "features",
    "models",
    "evaluation"
]

files = {
    "preprocessing/image_preprocessing.py": """import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    return resized
""",

    "preprocessing/ocr_extraction.py": """import pytesseract
from preprocessing.image_preprocessing import preprocess_image

def extract_text(image_path):
    image = preprocess_image(image_path)
    text = pytesseract.image_to_string(image)
    return text
""",

    "features/image_features.py": """from preprocessing.image_preprocessing import preprocess_image

def extract_image_features(image_path):
    image = preprocess_image(image_path)
    return image.flatten()
""",

    "features/text_features.py": """def extract_text_features(text):
    return len(text)
""",

    "models/train_model.py": """import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from features.image_features import extract_image_features
from features.text_features import extract_text_features
from preprocessing.ocr_extraction import extract_text

X, y = [], []

for label, folder in enumerate(["genuine", "fraudulent"]):
    path = os.path.join("dataset", folder)
    for file in os.listdir(path):
        image_path = os.path.join(path, file)

        img_feat = extract_image_features(image_path)
        text = extract_text(image_path)
        text_feat = extract_text_features(text)

        features = np.append(img_feat, text_feat)
        X.append(features)
        y.append(label)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "models/fraud_model.pkl")
print("Model trained successfully")
""",

    "evaluation/metrics.py": """from sklearn.metrics import accuracy_score, classification_report

def evaluate(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
""",

    "app.py": """import joblib
import numpy as np
from features.image_features import extract_image_features
from features.text_features import extract_text_features
from preprocessing.ocr_extraction import extract_text

model = joblib.load("models/fraud_model.pkl")

def predict_document(image_path):
    img_feat = extract_image_features(image_path)
    text = extract_text(image_path)
    text_feat = extract_text_features(text)

    features = np.append(img_feat, text_feat).reshape(1, -1)
    result = model.predict(features)

    return "Genuine Document" if result[0] == 0 else "Fraudulent Document"

if __name__ == "__main__":
    path = input("Enter document path: ")
    print(predict_document(path))
""",

    "requirements.txt": """opencv-python
pytesseract
numpy
scikit-learn
joblib
""",

    "README.md": """# Document Fraud Detection Using Machine Learning

This project detects fraudulent documents using image processing,
OCR, and machine learning.

Steps:
1. Add images to dataset/genuine and dataset/fraudulent
2. Run train_model.py
3. Run app.py
"""
}

# Create project folder
os.makedirs(project_name, exist_ok=True)
os.chdir(project_name)

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file_path, content in files.items():
    with open(file_path, "w") as f:
        f.write(content)

print("✅ Project structure created successfully!")
