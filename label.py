import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

# Paths
image_dir = "/home/jassem/dorra-labeling/preprocessed-images"  # Path to your images
annotations_csv = "/home/jassem/dorra-labeling/yolov5/annotations.csv"  # Path to your annotations CSV

# Load annotations
annotations = pd.read_csv(annotations_csv)

# Function to draw bounding boxes
def draw_bounding_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box[:4])
        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image

# Visualize bounding boxes for each image
output_dir = "/home/jassem/dorra-labeling/annotated-images"
os.makedirs(output_dir, exist_ok=True)

for _, row in annotations.iterrows():
    image_name = row['image_name']
    x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']

    image_path = os.path.join(image_dir, image_name)
    if os.path.exists(image_path):
        img_with_boxes = draw_bounding_boxes(image_path, [[x_min, y_min, x_max, y_max]])
        output_path = os.path.join(output_dir, f"annotated_{image_name}")
        cv2.imwrite(output_path, img_with_boxes)
        print(f"Saved: {output_path}")
    else:
        print(f"Image {image_name} not found!")
