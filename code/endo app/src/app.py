import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import threading
import http.server
import socketserver
import os
import webbrowser

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'Model/my_model.h5')

# Load your pre-trained model
model = tf.keras.models.load_model(model_path)

# Define your class names and descriptions
class_names = ['normal', 'ulcerative_colitis', 'polyps', 'esophagitis']
class_descriptions = {
    'normal': "The image shows normal, healthy tissue.",
    'ulcerative_colitis': "The image shows signs of ulcerative colitis, a chronic inflammatory bowel disease.",
    'polyps': "The image shows polyps, which are abnormal tissue growths that can develop in the colon or other areas.",
    'esophagitis': "The image shows esophagitis, an inflammation of the esophagus."
}

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
        return "No image uploaded", "", 0.0

    img_array = preprocess_image(img)

    prediction = model.predict(img_array)
    confidence = float(np.max(prediction, axis=1)[0])
    class_idx = np.argmax(prediction, axis=1)[0]

    if confidence < 0.3:  # Set your threshold here
        class_name = "unknown"
        class_description = "The confidence is too low to classify this image."
    else:
        class_name = class_names[class_idx]
        class_description = class_descriptions[class_name]

    return class_name, class_description, confidence

# Function to serve the welcome page
def serve_welcome_page():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", 8000), handler) as httpd:
        print("Serving at port 8000")
        webbrowser.open("http://localhost:8000/welcome.html")
        httpd.serve_forever()

# Define the Gradio interface with custom CSS
css = """
body { font-family: Arial, sans-serif; }
h1 { color: #4CAF50; }
.gradio-container { max-width: 800px; margin: 0 auto; }
"""

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Endoscopic Image"),  # Use "pil" type to ensure PIL Image format
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="Description"),
        gr.Number(label="Confidence")
    ],
    live=True,
    title="Endoscopic Image Classification - CHU Tanger",
    description="Upload an endoscopic image to classify it using our trained model.",
    theme="default",
    css=css
)

def launch_gradio():
    interface.launch(share=True)

# Run the welcome page in a separate thread
threading.Thread(target=serve_welcome_page).start()

# Launch the Gradio app
if __name__ == "__main__":
    launch_gradio()
