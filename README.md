🩺 Pneumonia Disease Predictor
A sleek Streamlit app powered by TensorFlow that classifies chest X-ray images as NORMAL or PNEUMONIA using a VGG16-based transfer learning model for fast and accurate single-image inference. [attached_file:1]

✨ Features
Simple web UI to upload a chest X-ray and obtain an immediate prediction with a confidence-style score derived from the model output. [attached_file:1]

Loads a pretrained Keras/TensorFlow model from an H5 file and performs preprocessing consistent with Keras utilities. [attached_file:1][web:46][web:48]

Transfer learning backbone based on VGG16, enabling efficient binary classification for chest X-rays. [attached_file:1]

🧩 Project Structure
text
.
├─ app.py                      # Streamlit entrypoint (rename if using a different filename)
├─ CNN-Transfer_VGG16.ipynb    # Training notebook (VGG16 transfer learning)
├─ Model/
│  └─ model_pretrained.h5      # Trained model loaded by the app
├─ requirements.txt            # Project dependencies
└─ README.md                   # Project documentation
The app expects the model at ./Model/model_pretrained.h5 as referenced in the frontend snippet. [attached_file:1]

The notebook trains a VGG16-based classifier and persists the model in H5 format for inference. [attached_file:1]

🛠️ Requirements
Use these versions for a smooth experience on Python 3.10+, or pin exact versions from the local environment with pip freeze for full reproducibility. [attached_file:1]

text
# Core ML/Deep Learning
tensorflow>=2.10.0

# Computer Vision
opencv-python>=4.5.0

# Data Processing & Scientific Computing
numpy>=1.21.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.5.0

# Optional but recommended
pillow>=9.0.0
scipy>=1.7.0
pandas>=1.4.0

# App framework
streamlit>=1.18.0

# Build support
setuptools>=65.0.0
wheel>=0.37.0
Streamlit is supported on modern Python versions and should be installed via pip in an isolated environment. [web:30]

🚀 Quickstart
Create a virtual environment and install dependencies from requirements.txt using the standard Streamlit installation flow. [web:30]

Ensure the trained model file exists at ./Model/model_pretrained.h5 as expected by the app frontend. [attached_file:1]

Launch the application with the Streamlit CLI from the project root. [web:41][web:44][web:47]

text
streamlit run app.py
🧪 Usage
Open the app in the browser and upload a chest X-ray image in JPEG format using the file uploader widget. [attached_file:1]

The app resizes the image to the configured target size (e.g., 100×100) and converts it to a NumPy array before prediction. [attached_file:1][web:48]

The model produces a binary output mapped to categories ["NORMAL", "PNEUMONIA"], and the UI displays the label with a score derived from the predicted value. [attached_file:1]

🧠 Model & Training
The notebook uses VGG16 as a feature extractor with a small classification head for binary classification on chest X-rays. [attached_file:1]

After training, the model is saved to H5 format, which can be loaded at inference time with tf.keras.models.load_model. [attached_file:1][web:46]

🐞 Troubleshooting
If loading the H5 model raises compatibility errors, verify the TensorFlow/Keras versions match training and use tf.keras.models.load_model for loading. [web:46]

For image IO utilities, prefer tf.keras.utils.load_img and tf.keras.utils.img_to_array in modern TensorFlow versions. [web:48]

Always start the server via the Streamlit CLI (streamlit run ...) to ensure proper app lifecycle and caching behavior. [web:41][web:44][web:47]

📦 Reproducibility Tips
To capture your exact environment, generate a fully pinned requirements file after testing the app locally. [attached_file:1]

text
pip freeze > requirements.txt
Keep the model file under version control with Git LFS if the H5 exceeds typical repository limits, or store it in a model registry or artifact store. [attached_file:1]

🧭 Notes
The current frontend targets JPEG uploads; extend the file_uploader types to include PNG if needed for broader compatibility. [attached_file:1]

If migrating between TensorFlow versions, validate that saved model formats and custom objects are handled consistently during load. [web:36][web:46]

🙌 Acknowledgments
Built with TensorFlow Keras APIs for saving and loading models and with Streamlit for interactive model inference. [web:46][web:41]