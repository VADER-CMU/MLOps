import numpy as np
import cv2
import random
from shutil import copy
from pathlib import Path

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import cv2

img_path = "/home/kshitij/Documents/Bell Pepper/dataset-collection/dataset_620_red_yellow_cart_only/_Color_1607625615060.01098632812500.png"
image = plt.imread(img_path)
annotation_path = "/home/kshitij/Documents/Bell Pepper/dataset-collection/labels/_Color_1607625615060.01098632812500.txt"

yolo_annotations = open(annotation_path).read()

fig, ax = plt.subplots(figsize=(10,10))

height = image.shape[0]
width = image.shape[1]

image_shape = np.array([width, height]).reshape((-1,2))

annotation_poly = []
fruit_poly = []
peduncle_poly = []

PEDUNCLE_IMGE_SIZE = 640

for line in yolo_annotations.split("\n"):
    # y = np.array(line.split(" ")[1:]).astype(np.float32)
    if line.split(" ")[0] == '0':
        fruit_y = np.array(line.split(" ")[1:]).astype(np.float32)
        fruit_poly.append(fruit_y)
    if line.split(" ")[0] == '1':
        peduncle_y = np.array(line.split(" ")[1:]).astype(np.float32)
        peduncle_poly.append(peduncle_y)

for poly in fruit_poly[0:1]:
    x_coords = poly.reshape((-1,2))[:, 0] * width
    y_coords = poly.reshape((-1,2))[:, 1] * height

    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    max_size = max(x_max-x_min, y_max-y_min)

    roi_y_min = max(0, (y_min - max_size//2).astype(np.int32))
    roi_y_max = min(height, (y_max + max_size//2).astype(np.int32))
    roi_x_min = max(0, (x_min - max_size//2).astype(np.int32))
    roi_x_max = min(width, (x_max + max_size//2).astype(np.int32))

    cropped_image = image[roi_y_min:roi_y_max, roi_x_min:roi_x_max, :]
    
    cropped_image = cv2.resize(cropped_image, (PEDUNCLE_IMGE_SIZE, PEDUNCLE_IMGE_SIZE))
    roi_peduncle_poly = None

    for poly in peduncle_poly:
        scaled_poly = poly.reshape((-1,2)) * image_shape

        if (scaled_poly[:, 0].min() >= roi_x_min and scaled_poly[:, 0].max() <= roi_x_max and
            scaled_poly[:, 1].min() >= roi_y_min and scaled_poly[:, 1].max() <= roi_y_max):

            roi_peduncle_poly = scaled_poly

    new_annotation = []
    for point in roi_peduncle_poly.reshape(-1, 2):
        # Normalize coordinates to [0, 1]
        x_norm = (point[0] - roi_x_min) / (roi_x_max - roi_x_min)
        y_norm = (point[1] - roi_y_min) / (roi_y_max - roi_y_min)
        
        # Add to YOLO line
        new_annotation.append(f"{x_norm:.6f} {y_norm:.6f}")


    new_annotation_line = "0 " + " ".join(new_annotation)

ax.imshow(cropped_image)
roi_peduncle_poly = []
print("New Annotation Line:", new_annotation_line)
for line in new_annotation_line.split("\n"):
    # y = np.array(line.split(" ")[1:]).astype(np.float32)
    if line.split(" ")[0] == '0':
        roi_peduncle_y = np.array(line.split(" ")[1:]).astype(np.float32)
        roi_peduncle_poly.append(roi_peduncle_y)

for poly in roi_peduncle_poly:
    scaled_poly = poly.reshape((-1,2)) * cropped_image.shape[0:2][::-1]
    p = Polygon(scaled_poly, facecolor = 'green', edgecolor = 'g',  fill = True, alpha=0.5)

    ax.add_patch(p)

plt.show()