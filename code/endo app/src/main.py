# classify/gradio_interface.py
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your pre-trained model
model = tf.keras.models.load_model('Model/my_model.h5')

# Define your class names
class_names = ['0_normal', '1_ulcerative_colitis', '2_polyps', '3_esophagitis']

# Define the scalar function used in training
def scalar(img):
    return img

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert("RGB")  # Ensure image is in RGB mode
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = scalar(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def classify_image(img):
    if img is None:
        return "No image uploaded"

    img_array = preprocess_image(img)

    # Debug: Print the shape of the input array
    print("Shape of preprocessed image array:", img_array.shape)

    prediction = model.predict(img_array)

    # Debug: Print the raw prediction
    print("Raw prediction:", prediction)

    class_idx = np.argmax(prediction, axis=1)[0]
    class_name = class_names[class_idx]

    # Debug: Print the predicted class index and name
    print(f"Predicted class index: {class_idx}, Predicted class name: {class_name}")

    return f"Class: {class_name}"

# Define the Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Image"),  # Use "pil" type to ensure PIL Image format
    outputs=gr.Label(label="Prediction"),
    live=True,
    title="Endoscopic Image Classification",
    description="Upload an endoscopic image to classify it using the trained model.",
    theme="default"
)

def launch_gradio():
    interface.launch(share=True)

if __name__ == "__main__":
    launch_gradio()
