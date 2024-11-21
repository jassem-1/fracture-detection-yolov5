import pandas as pd
import os
import cv2

# Paths
annotations_csv = "/home/jassem/dorra-labeling/yolov5/annotations.csv"  # Original CSV
image_dir = "/home/jassem/dorra-labeling/preprocessed-images"  # Path to images
labels_dir = "/home/jassem/dorra-labeling/yolo_data/labels"  # Output directory for YOLO labels

# Create labels directory if it doesn't exist
os.makedirs(labels_dir, exist_ok=True)

# Load annotations
annotations = pd.read_csv(annotations_csv)

# Normalize and correct bounding boxes
for _, row in annotations.iterrows():
    image_name = row['image_name']
    x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
    class_id = 0  # Assuming one class: fracture

    # Load image to get dimensions
    image_path = os.path.join(image_dir, image_name)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        continue
    h, w = img.shape[:2]

    # Normalize bounding box coordinates
    x_center = (x_min + x_max) / 2 / w
    y_center = (y_min + y_max) / 2 / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h

    # Skip invalid bounding boxes
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        print(f"Skipping invalid bounding box for {image_name}: {x_center}, {y_center}, {width}, {height}")
        continue

    # Write to YOLO label file
    label_file = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    with open(label_file, "w") as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("Bounding box normalization complete.")
