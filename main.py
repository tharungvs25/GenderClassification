!pip install transformers
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, AutoImageProcessor
from PIL import Image
import cv2
import os
from google.colab.patches import cv2_imshow
import numpy as np
from io import BytesIO  # Import BytesIO
from google.colab import files  # Import files

# Load gender classification model and feature extractor (Do this ONCE at the top)
processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification-2")
model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification-2")
feature_extractor = AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification-2")

def detect_faces(image_path):
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Error: Could not load face detection classifier.")
        return np.array([]), None

    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image at '{image_path}'")
        return np.array([]), None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return faces, img

def classify_gender(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()  # Correct indentation
    labels = model.config.id2label
    predicted_label = labels[predicted_class_idx]
    return predicted_label

# Upload image
uploaded = files.upload()

for filename, file_content in uploaded.items(): # Corrected loop
    image = Image.open(BytesIO(file_content)).convert("RGB")
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(file_content)

    faces, image_with_boxes = detect_faces(image_path)

    if len(faces) > 0:  # Correct face check
        for (x, y, w, h) in faces:
            face_roi = image.crop((x, y, x + w, y + h))
            predicted_gender = classify_gender(face_roi)

            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, predicted_gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2_imshow(image_with_boxes)
        cv2.imwrite("faces_with_gender.jpg", cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    else:
        print("No faces found in the image.")

    os.remove(image_path)
