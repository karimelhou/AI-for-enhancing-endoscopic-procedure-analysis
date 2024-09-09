import cv2
import numpy as np
import os

# Define color ranges (in HSV) for bounding box colors
# Adjusted ranges to encompass variations in yellow color
bbox_colors = [
    {"name": "yellow", "lower": np.array([20, 100, 100]), "upper": np.array([40, 255, 255])},
    {"name": "black", "lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 30])}
]

input_directory = "detection des lesions/train"
output_directory = "detection des lesions output/train"

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to apply Non-Maximum Suppression
def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []
    return [boxes[i[0]] for i in indices]

# Function to extract bounding boxes and visualize them
def extract_and_visualize_bounding_boxes(image_path, bbox_colors, min_area=500, max_area=10000):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    kernel = np.ones((5, 5), np.uint8)  # Kernel for morphological operations
    all_boxes = []
    all_scores = []
    for bbox_color in bbox_colors:
        mask = cv2.inRange(hsv_image, bbox_color["lower"], bbox_color["upper"])
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)

        # Debugging: Save mask
        mask_output_path = os.path.join(output_directory, f'{os.path.splitext(os.path.basename(image_path))[0]}_{bbox_color["name"]}_mask.png')
        cv2.imwrite(mask_output_path, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            # Ignore contours that are too small or too large
            if area < min_area or area > max_area:
                continue
            # Ignore contours that are likely on the edge of the image
            if x == 0 or y == 0 or x + w == image.shape[1] or y + h == image.shape[0]:
                continue
            all_boxes.append([x, y, w, h])
            all_scores.append(1.0)  # Assign a dummy score of 1.0 to each box

    if not all_boxes:
        return image  # Return the image as is if no valid boxes are found

    # Apply Non-Maximum Suppression
    try:
        nms_boxes = non_max_suppression(all_boxes, all_scores, iou_threshold=0.1)
    except Exception as e:
        print(f"Error during NMS: {e}")
        return image

    for box in nms_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "yellow", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Process each class folder
for class_name in ["polyps", "ulcerative colitis", "normal"]:
    class_input_directory = os.path.join(input_directory, class_name)
    class_output_directory = os.path.join(output_directory, class_name)

    if not os.path.exists(class_output_directory):
        os.makedirs(class_output_directory)

    for image_name in os.listdir(class_input_directory):
        if image_name.endswith(".png") or image_name.endswith(".jpg"):
            image_path = os.path.join(class_input_directory, image_name)
            output_image_path = os.path.join(class_output_directory, image_name)

            print(f"Processing image: {image_path}")
            visualized_image = extract_and_visualize_bounding_boxes(image_path, bbox_colors)

            if visualized_image is not None:
                cv2.imwrite(output_image_path, visualized_image)
                print(f"Saved visualized image: {output_image_path}")
            else:
                print(f"No valid contours found for image: {image_path}")

print("Bounding box visualization completed.")
