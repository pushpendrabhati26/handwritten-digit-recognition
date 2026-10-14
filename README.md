# Handwritten Digit Recognition

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features
- Draw a digit on the canvas.
- Predict the digit using a trained CNN model.
- Display prediction confidence and scores for all digits.
- View the processed image sent to the model.

## Technologies Used
- Python
- TensorFlow and Keras
- Streamlit
- NumPy
- Pillow
- MNIST Dataset

## Run Locally
Install the dependencies:

`pip install -r requirements.txt`

Start the application:

`streamlit run app.py`

## Model
The trained model is saved as `digit_model.keras`.