import os

# Path to the labels folder
labels_dir = "/home/jassem/dorra-labeling/dataset-yolov/train/labels"

# Find all unique class IDs
class_ids = set()
for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):
        with open(os.path.join(labels_dir, label_file), "r") as f:
            for line in f:
                class_id = int(line.split()[0])  # Get the first value (class_id)
                class_ids.add(class_id)

# Print results
print(f"Number of Classes (nc): {len(class_ids)}")
print(f"Class IDs: {sorted(class_ids)}")
