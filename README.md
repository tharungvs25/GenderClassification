# Gender-Classifier: Face Detection and Gender Classification

Gender-Classifier is a simple yet effective system that combines face detection (using OpenCV's Haar Cascades) with deep learning (Hugging Face Transformers) to classify the gender of individuals in images. The project leverages pre-trained models for rapid deployment and demonstrates a clear pipeline for image analysis.

---

## 📂 Project Structure
```
Gender_Classifier/
|
├── models/
│   └── rizvandwiki/gender-classification-2/ # Pre-trained Hugging Face model files
|
└── gender_classifier.ipynb                 # Main pipeline notebook (or .py script)
```
## 📊 Dataset & Labeling (Conceptual for Pre-trained Model)

This project primarily utilizes a pre-trained model, meaning the dataset and labeling were handled by the original model creators.

- **Pre-trained Model Dataset:** The `rizvandwiki/gender-classification-2` model was trained on a diverse dataset of images labeled for gender.
- **Classes:**
  - `Female`
  - `Male`

## 🚀 Pipeline Overview

### 1. Face Detection (OpenCV)

- **Method:** Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
- **Task:** Detects bounding boxes for human faces in an input image.
- **Output:** Coordinates `(x, y, w, h)` for each detected face.

### 2. Gender Classification (Hugging Face Transformers)

- **Model:** `rizvandwiki/gender-classification-2` (Vision Transformer based)
- **Task:** Image classification on the cropped face regions.
- **Input:** Cropped image region corresponding to a detected face.
- **Output:** Predicted gender label (`Male` or `Female`).

## 🤝 Inference Flow
```
[User Image Upload (via Google Colab files.upload() or local file)]
⬇
[Image Loaded & Preprocessed]
⬇
[Face Detection (OpenCV)]
⬇
[For each detected face:]
⬇
[Crop Face Region]
⬇
[Gender Classification (Hugging Face Model)]
⬇
[Annotate Image with Bounding Box & Gender Label]
⬇
[Display Annotated Image (via cv2_imshow or local display)]
⬇
[Save Annotated Image]
```
---

## 🔄 Performance (of the Pre-trained Model)

The performance metrics provided here refer to the underlying `rizvandwiki/gender-classification-2` model, as its accuracy directly impacts the project's classification results.

| Metric | Value (Approx.) |
|--------|-----------------|
| Accuracy | High (as reported by model creators on Hugging Face) |

*Note: Specific metrics like precision, recall, and F1-score for this particular model would typically be found on its Hugging Face model card.*

---

## 📆 Future Work

- **Portability:** Refactor the code to be easily runnable outside of Google Colab, allowing for standard local execution or integration into web/desktop applications.
- **Real-time Processing:** Adapt the pipeline for live video stream processing (e.g., from a webcam).
- **Batch Processing:** Implement functionality to process multiple images from a directory.
- **Error Handling:** Enhance error handling for cases like no faces detected, image loading failures, or model issues.
- **Confidence Scores:** Display confidence scores for the gender predictions.
- **Alternative Face Detectors:** Explore more advanced face detection models (e.g., MTCNN, FaceNet, or other deep learning-based detectors) for improved accuracy and robustness.
- **GUI/Web Interface:** Develop a simple graphical user interface or a basic web application (using Flask/Streamlit) for easier interaction.

---

## 📄 Tech Stack

- Python
- `transformers` (Hugging Face)
- `torch`
- `opencv-python` (OpenCV)
- `Pillow` (PIL)
- `numpy`
- Google Colab (for current usage environment)
