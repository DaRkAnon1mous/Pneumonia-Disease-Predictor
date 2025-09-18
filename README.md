# Pneumonia Detection using CNN and Transfer Learning

![Project Banner](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![License](https://img.shields.io/badge/License-MIT-green)

This project implements a Convolutional Neural Network (CNN) model using transfer learning with the VGG16 architecture to classify chest X-ray images as either **NORMAL** or **PNEUMONIA**. The model is trained on a dataset of chest X-rays and achieves high accuracy in detecting pneumonia. Additionally, a user-friendly web app built with Streamlit allows users to upload X-ray images for real-time predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Jupyter Notebook](#running-the-jupyter-notebook)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Pneumonia is a serious lung infection that can be detected early through chest X-rays. This project automates the detection process using deep learning:
- **Transfer Learning**: Utilizes a pre-trained VGG16 model (from Keras) as the base, with additional dense layers for binary classification.
- **Data Augmentation**: Applied using `ImageDataGenerator` to improve model generalization.
- **Deployment**: A Streamlit web app for easy image uploads and predictions.
- **Key Libraries**: TensorFlow/Keras for model building, OpenCV and Matplotlib for image processing, Streamlit for the UI.

The model is fine-tuned on grayscale-converted X-ray images resized to 100x100 pixels.

## Dataset
The dataset consists of chest X-ray images divided into two classes: **NORMAL** and **PNEUMONIA**. It is organized into three directories:
- `train/`: 5,226 images for training.
- `val/`: 16 images for validation.
- `test/`: 624 images for testing.

Source: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Images are loaded in grayscale mode using OpenCV, as X-rays are monochromatic, reducing computational complexity.

## Model Architecture
- **Base Model**: VGG16 (pre-trained on ImageNet), with the top layers removed.
- **Custom Layers**:
  - Flatten layer.
  - Dense(256, activation='relu').
  - Dense(128, activation='relu').
  - Dense(64, activation='relu').
  - Dense(1, activation='sigmoid') for binary classification.
- **Total Parameters**: ~18.4 million (trainable: ~1.2 million).
- **Optimizer**: Adam.
- **Loss**: Binary Crossentropy.
- **Training**: 10 epochs with data augmentation (shear, zoom, horizontal flip).
- **Input Shape**: (100, 100, 3) – images are converted to RGB for VGG16 compatibility.

For full details, refer to the [Jupyter Notebook](CNN+Transfer_VGG16.ipynb).

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/pneumonia-detection.git
   cd pneumonia-detection
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Create `requirements.txt` with: tensorflow, keras, streamlit, opencv-python, numpy, pandas, matplotlib)

   Or manually:
   ```
   pip install tensorflow keras streamlit opencv-python numpy pandas matplotlib
   ```
3. Download the pre-trained model weights: Place `model_pretrained.h5` in the `Model/` directory (or train the model yourself using the notebook).

## Usage

### Running the Jupyter Notebook
1. Open `CNN+Transfer_VGG16.ipynb` in Jupyter:
   ```
   jupyter notebook CNN+Transfer_VGG16.ipynb
   ```
2. Run cells sequentially to:
   - Load and preprocess data.
   - Build and train the model.
   - Evaluate on test data.
   - Save the model as `model_pretrained.h5`.

### Running the Streamlit App
1. Start the app:
   ```
   streamlit run app.py
   ```
2. Upload a chest X-ray image (JPEG format).
3. Click **PREDICT** to get the result (e.g., "PNEUMONIA - 99.87%").

   Example Output:
   - Displays the uploaded image.
   - Predicts the class with confidence score.

## Results
- **Training Accuracy**: ~95-96% after 10 epochs.
- **Validation Accuracy**: ~93.75%.
- **Test Accuracy**: 89.10%.
- **Loss**: Converges to ~0.10 on training, ~0.20 on validation.

The model performs well but may overfit slightly; Visualizations of sample images and model summary are in the notebook.

<img width="1851" height="1033" alt="image" src="https://github.com/user-attachments/assets/b38627ff-bf55-4281-826b-b9ddff96fbf5" />
<img width="660" height="917" alt="image" src="https://github.com/user-attachments/assets/4a402517-6846-4252-878c-0cd3fbbe0ea0" />



## Contributing
Contributions are welcome! Fork the repo, create a branch, and submit a pull request. For major changes, open an issue first.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
