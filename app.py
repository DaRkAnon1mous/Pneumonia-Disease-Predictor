import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .title {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 20px;
    }
    .prediction-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #135d93;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
    }
    </style>
""", unsafe_allow_html=True)

# Model and constants
PRETRAINED_MODEL = "./Model/model_pretrained.h5"
CATEGORIES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 100
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(PRETRAINED_MODEL)

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}. Please ensure 'model_pretrained.h5' is in the 'Model/' directory.")
    st.stop()

# Sidebar content
st.sidebar.title("About")
st.sidebar.markdown("""
This app uses a **VGG16-based CNN** to classify chest X-ray images as **NORMAL** or **PNEUMONIA**. Upload a chest X-ray image (JPEG, <10MB) to get a prediction with confidence score.

**Note**:
- Ensure the image is a clear chest X-ray.
- Predictions are for demonstration purposes; consult a medical professional for accurate diagnosis.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using **Streamlit** and **TensorFlow**")

# Main content
st.markdown('<div class="title">Pneumonia Detection from Chest X-Rays</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload a chest X-ray image to detect pneumonia</div>', unsafe_allow_html=True)

def load_classifier():
    # File uploader
    uploaded_file = st.file_uploader(
        label="Choose an X-ray image (JPEG)",
        type=['jpeg', 'jpg'],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("File size exceeds 10MB. Please upload a smaller file.")
            return

        try:
            # Load and process image
            img = Image.open(uploaded_file)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            
            # Convert grayscale to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            st.image(img, caption="Uploaded X-ray Image", use_container_width=True)

            # Convert image to array
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0

            # Prediction button
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Predict"):
                    with st.spinner("Making prediction..."):
                        # Make prediction
                        prediction = model.predict(img_array)
                        confidence = prediction[0][0]
                        predicted_class = CATEGORIES[int(round(confidence))]
                        confidence_percent = round(confidence * 100, 2) if predicted_class == "PNEUMONIA" else round((1 - confidence) * 100, 2)
                        result = f"{predicted_class} - {confidence_percent}%"
                        
                        # Display result
                        st.markdown(f'<div class="prediction-text">Prediction: {result}</div>', unsafe_allow_html=True)

            # Reset button
            with col2:
                if st.button("Reset"):
                    st.session_state.clear()
                    st.rerun()

        except Exception as e:
            st.error(f"Error processing image: {str(e)}. Please ensure it's a valid JPEG image.")

def main():
    load_classifier()

if __name__ == "__main__":
    main()