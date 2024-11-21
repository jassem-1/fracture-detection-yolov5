import os
import cv2

# Define the directories
image_dir = "/home/jassem/dorra-labeling/dataset-yolov/train/images"  # Path to your images
labels_dir = "/home/jassem/dorra-labeling/dataset-yolov/train/labels"  # Path to YOLO labels
output_dir = "/home/jassem/dorra-labeling/boxes-verif1"  # Folder to save images with boxes

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to visualize YOLO labels and save images
def visualize_and_save_label(image_name):
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    output_path = os.path.join(output_dir, image_name)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return

    h, w = img.shape[:2]

    # Read the labels
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return

    with open(label_path, "r") as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            # Convert YOLO format to pixel coordinates
            x_min = int((x_center - width / 2) * w)
            y_min = int((y_center - height / 2) * h)
            x_max = int((x_center + width / 2) * w)
            y_max = int((y_center + height / 2) * h)
            # Draw bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Save the image with bounding boxes
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

# Visualize and save bounding boxes for all images
for image_name in os.listdir(image_dir):
    visualize_and_save_label(image_name)
