import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Define custom objects
smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred, smooth=1e-5):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

# Load the pre-trained model
model_path = 'model/module.keras'
model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'iou': iou, 'dice_coef': dice_coef})

# Define preprocessing and postprocessing functions
IMG_H = 256
IMG_W = 256

def preprocess(image):
    image = cv2.resize(image, (IMG_W, IMG_H))
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postprocess(mask):
    mask = mask[0, :, :, 0]  # Remove batch dimension and channel dimension
    mask = (mask > 0.5).astype(np.uint8) * 255  # Binarize mask
    return mask

def predict(image):
    image = preprocess(image)
    mask = model.predict(image)
    mask = postprocess(mask)
    return mask

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    description="Upload an image and the model will generate the segmentation mask."
)

# Launch the Gradio interface
iface.launch()
