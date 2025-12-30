import cv2
import os
import numpy as np
import pytesseract
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load images and labels
def load_data(folder):
    X, y = [], []
    for label, sub in enumerate(["genuine", "fraudulent"]):
        path = os.path.join(folder, sub)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))
            features = image.flatten()

            text = pytesseract.image_to_string(image)
            features = np.append(features, len(text))

            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Load dataset
X, y = load_data("dataset")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Predict new document
def predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    features = img.flatten()
    text = pytesseract.image_to_string(img)
    features = np.append(features, len(text)).reshape(1, -1)

    result = model.predict(features)
    return "Genuine" if result[0] == 0 else "Fraudulent"

# Example
print(predict("dataset/genuine/sample.jpg"))
