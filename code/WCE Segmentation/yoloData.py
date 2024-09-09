import cv2
import numpy as np
import os


def convert_to_yolo_format(image_width, image_height, x_min, y_min, x_max, y_max, class_id):
    center_x = (x_min + x_max) / 2.0 / image_width
    center_y = (y_min + y_max) / 2.0 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return f"{class_id} {center_x} {center_y} {width} {height}"


def extract_bounding_boxes(image_path, bbox_color_lower, bbox_color_upper, class_id):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, bbox_color_lower, bbox_color_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width, _ = image.shape
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        yolo_format = convert_to_yolo_format(width, height, x, y, x + w, y + h, class_id)
        bboxes.append(yolo_format)

    return bboxes


def save_yolo_annotations(output_directory, image_name, bboxes):
    annotation_file = os.path.join(output_directory, f"{os.path.splitext(image_name)[0]}.txt")
    with open(annotation_file, "w") as f:
        for bbox in bboxes:
            f.write(bbox + "\n")


# Define color ranges (in HSV) for bounding box colors
bbox_colors = [
    {"name": "polyps", "class_id": 0, "lower": np.array([75, 150, 150]), "upper": np.array([90, 255, 255])},
    {"name": "ulcers", "class_id": 1, "lower": np.array([75, 150, 150]), "upper": np.array([90, 255, 255])}
]

input_directory = "detection_des_lesions"
output_directory = "yolo_annotations"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for bbox_color in bbox_colors:
    class_name = bbox_color["name"]
    class_id = bbox_color["class_id"]
    bbox_color_lower = bbox_color["lower"]
    bbox_color_upper = bbox_color["upper"]

    class_input_directory = os.path.join(input_directory, class_name)
    class_output_directory = os.path.join(output_directory, class_name)

    if not os.path.exists(class_output_directory):
        os.makedirs(class_output_directory)

    for image_name in os.listdir(class_input_directory):
        if image_name.endswith(".png") or image_name.endswith(".jpg"):
            image_path = os.path.join(class_input_directory, image_name)
            bboxes = extract_bounding_boxes(image_path, bbox_color_lower, bbox_color_upper, class_id)
            save_yolo_annotations(class_output_directory, image_name, bboxes)

print("Conversion to YOLO format completed.")
