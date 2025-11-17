import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Glaucoma Detector",
    layout="centered"
)

MODEL_PATH = "glaucoma_model.h5"

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_glaucoma_model():
    model = load_model(MODEL_PATH)
    return model

model = load_glaucoma_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = np.array(image)

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Resize to model input size (224x224)
    image = cv2.resize(image, (224, 224))

    # Normalize
    image = image / 255.0

    # Expand dims -> (1,224,224,3)
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    # 0 → normal, 1 → glaucoma
    label = "Glaucoma" if prediction >= 0.5 else "Normal"
    confidence = float(prediction if prediction >= 0.5 else 1 - prediction)

    return label, confidence

# -----------------------------
# UI
# -----------------------------
st.title("👁️ Glaucoma Detection App")
st.write("Upload a retinal image to detect if it is **Glaucoma** or **Normal**.")

uploaded_file = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            label, confidence = predict(image)

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")
