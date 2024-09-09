import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define your class names and descriptions
class_names = ['0_normal', '1_ulcerative_colitis', '2_polyps', '3_esophagitis']

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
def classify_image(model, img_path):
    img = Image.open(img_path)
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    confidence = float(np.max(prediction, axis=1)[0])
    class_idx = np.argmax(prediction, axis=1)[0]
    class_name = class_names[class_idx]
    return class_name, confidence

# Load the model
model_path = r'src\Model\my_model.h5'
model = tf.keras.models.load_model(model_path)

# Loop over the dataset and make predictions
test_dir = r'C:\Users\elhou\OneDrive\Documents\PFE\Pycharm Projects\NN\WCE Anomaly Detection\dataset\test'
true_labels = []
predicted_labels = []

for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        predicted_class, confidence = classify_image(model, img_path)
        true_labels.append(class_name)
        predicted_labels.append(predicted_class)

# Convert true and predicted labels to numerical format
true_labels_num = [class_names.index(label) for label in true_labels]
predicted_labels_num = [class_names.index(label) for label in predicted_labels]

# Calculate metrics
accuracy = accuracy_score(true_labels_num, predicted_labels_num)
print(f"Accuracy: {accuracy}")
print(classification_report(true_labels_num, predicted_labels_num, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(true_labels_num, predicted_labels_num)

# Plotting Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("Actual Classes")
plt.xlabel("Predicted Classes")
plt.savefig('confusion_matrix.png')  # Save the plot as a .png file
plt.show()

# Plotting Precision, Recall, and F1-Score
classification_rep = classification_report(true_labels_num, predicted_labels_num, target_names=class_names, output_dict=True)
precision = [classification_rep[class_name]['precision'] for class_name in class_names]
recall = [classification_rep[class_name]['recall'] for class_name in class_names]
f1_score = [classification_rep[class_name]['f1-score'] for class_name in class_names]

x = np.arange(len(class_names))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Classes')
ax.set_title('Precision, Recall, and F1-Score by Class')
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend()

fig.tight_layout()

plt.savefig('precision_recall_f1_score.png')  # Save the plot as a .png file
plt.show()
