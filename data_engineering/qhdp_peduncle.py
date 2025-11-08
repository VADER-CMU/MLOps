import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pathlib
import albumentations as A
from pathlib import Path
import json
from tqdm import tqdm

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def mask_to_yolo_format(mask, classes_json_path="classes.json", group_by_type=True):
    """
    Convert an instance segmentation mask to YOLO format using hex colors from JSON
    
    Args:
        mask: numpy array (H, W, 3) with RGB values corresponding to class colors
        classes_json_path: path to JSON file with class definitions and hex colors
        group_by_type: if True, all peppers get class_id=0, all peduncles get class_id=1
                      if False, each class gets its own ID (0-indexed)
        
    Returns:
        string of YOLO format annotations (class_id, x1, y1, x2, y2, ...)
    """
    # Load class definitions
    with open(classes_json_path, 'r') as f:
        classes_data = json.load(f)
    
    # Create color to class ID mapping
    color_to_class = {}
    for cls in classes_data['classes']:
        rgb = hex_to_rgb(cls['color'])
        if group_by_type:
            # Group by type: peppers = 0, peduncles = 1
            if 'peduncle' in cls['name'].lower():
                color_to_class[rgb] = 0
        else:
            color_to_class[rgb] = cls['id'] - 1  # Convert to 0-indexed for YOLO
    
    height, width = mask.shape[:2]
    yolo_annotations = []
    
    # Find unique colors in the mask
    if len(mask.shape) == 2:
        # Grayscale mask - treat as instance IDs
        unique_colors = [(val,) for val in np.unique(mask) if val > 0]
    else:
        # RGB mask
        mask_reshaped = mask.reshape(-1, 3)
        unique_colors = np.unique(mask_reshaped, axis=0)
        # Filter out black background
        unique_colors = [tuple(color) for color in unique_colors 
                        if not np.all(color == 0)]
    
    for color in unique_colors:
        # Get class ID for this color
        if color not in color_to_class:
            continue
        
        class_id = color_to_class[color]
        
        # Create binary mask for this color
        if len(mask.shape) == 2:
            instance_mask = (mask == color[0]).astype(np.uint8)
        else:
            instance_mask = np.all(mask == color, axis=-1).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        if len(contours) > 1:
            return None
        # Process each contour (multiple instances of same class)
        for contour in contours:
            # Skip small contours
            if cv2.contourArea(contour) < 10:
                continue
            
            # Approximate contour
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Need at least 3 points to define a polygon
            if len(approx) < 3:
                continue
            
            yolo_line = f"{class_id}"
            
            for point in approx.reshape(-1, 2):
                # Normalize coordinates to [0, 1]
                x_norm = point[0] / width
                y_norm = point[1] / height
                
                # Add to YOLO line
                yolo_line += f" {x_norm:.6f} {y_norm:.6f}"
            
            yolo_annotations.append(yolo_line)
    
    yolo_string = "\n".join(yolo_annotations)
    return yolo_string

PEDUNCLE_IMAGE_SIZE = 640

def extract_cropped_regions(xml_path, image_path):
    """
    Parses a PASCAL VOC XML file and extracts the regions of interest from the specified image.

    Args:
        xml_path (str): Filepath to the XML annotation file.
        image_path (str): Filepath to the image.
    """
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image = cv2.imread(image_path)
        
    size_node = root.find('size')
    width = int(size_node.find('width').text)
    height = int(size_node.find('height').text)

    # --- 3. Find all objects and store their boxes ---
    boxes_to_extract = []
    for obj in root.findall('object'):
        # name = obj.find('name').text # No longer needed
        bndbox = obj.find('bndbox')
        
        # Get coordinates and convert from string to integer
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        boxes_to_extract.append((xmin, ymin, xmax, ymax))


    # Get dimensions for a reasonable plot size
    height, width, _ = image.shape
    
    cropped_images = []
    # --- Draw boxes using plt.Rectangle ---
    for box in boxes_to_extract:
        xmin, ymin, xmax, ymax = box

        max_size = max(xmax - xmin, ymax - ymin)

        roi_y_min = max(0, (ymin - max_size//2))
        roi_y_max = min(height, (ymax + max_size//2))
        roi_x_min = max(0, (xmin - max_size//2))
        roi_x_max = min(width, (xmax + max_size//2))

        cropped_image = image[roi_y_min:roi_y_max, roi_x_min:roi_x_max, :]
        
        cropped_image = (cv2.resize(cropped_image, (PEDUNCLE_IMAGE_SIZE, PEDUNCLE_IMAGE_SIZE))).astype(np.uint8)
        cropped_images.append(cropped_image)

    return cropped_images

# Define the augmentation pipeline
def create_train_augmentation_pipeline():
    transform = A.Compose([
        # Apply random rotation (up to 60 degrees) and translation
        # A.PadIfNeeded(min_height=1000, min_width=1000, border_mode=cv2.BORDER_CONSTANT, value=0),

        # Apply random rotation up to 45 degrees
        A.HorizontalFlip(p=0.5),
        # Always rotate by 90 degrees
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(90, 90), p=1.0),
        

        A.Rotate(limit=10, p=0.1),  # p=1.0 ensures rotation is always applied[3]
        A.Affine(scale=(1, 1.1), p=0.3 ),
        A.RandomBrightnessContrast(
        brightness_limit=0.2,     # Handle darker/brighter conditions
        contrast_limit=0.2,
        p=0.4
    ),
    A.RandomGamma(
        gamma_limit=(70, 110),    # Gamma correction for different exposures
        p=0.4
    ),

    ])
    return transform


if __name__ == "__main__":
    # xml_file = "annotations.xml"
    # xml_path = pathlib.Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/qhdp2020/QHDP_2020/train/2017_07_30_row1a_1501399080022352104__realsense_rgb_image_raw.xml")
    # xml_file = str(xml_path)

    # tree = ET.parse(xml_file)
    # root = tree.getroot()
    # image_name_from_xml = xml_path.parent / root.find('filename').text

    # image_file = image_name_from_xml

    # img = cv2.imread(str(image_file))
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    # cropped_regions = extract_cropped_regions(xml_file, image_file)

    # for img in cropped_regions:
    #     fig, ax = plt.subplots(figsize=(10,10))
    #     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     plt.show()

    # peduncle_folder = pathlib.Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/qhdp2020/Peduncle_QHDP/train/peduncle_imgs")
    # cropped_folder = pathlib.Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/qhdp2020/Peduncle_QHDP/train/cropped_imgs")

    # xml_list = list(peduncle_folder.glob("*.xml"))
    # for xml_file in xml_list:
    #     tree = ET.parse(str(xml_file))
    #     root = tree.getroot()
    #     image_name_from_xml = xml_file.parent / root.find('filename').text

    #     image_file = image_name_from_xml

    #     cropped_regions = extract_cropped_regions(xml_file, image_file)
    #     counter = 0

    #     for img in cropped_regions:
    #         cv2.imwrite(os.path.join(cropped_folder, os.path.basename(xml_file).replace('.xml', f'_peduncle{counter}.png')), img)
    #         counter += 1
    qhdp_images = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/qhdp2020/Peduncle_QHDP/train/cropped_imgs/images")
    qhdp_masks = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/qhdp2020/Peduncle_QHDP/train/cropped_imgs/labels")
    peduncle_mega_val = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/peduncle_mega/val")
    img_list = list(qhdp_images.glob("*.png"))

    num_augmentations = 3

    for img_path in tqdm(img_list, desc="Processing images"):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(qhdp_masks / img_path.stem)+'.png', cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        for i in range(num_augmentations):
            transform = create_train_augmentation_pipeline()

            transformed = transform(image=image, masks=[mask])

            aug_image = transformed['image']
            aug_mask = transformed['masks'][0]

            # # Save augmented images and masks
            # 
            annotations = mask_to_yolo_format(aug_mask, classes_json_path=str(qhdp_images.parent / "classes.json"))
            if annotations is None:
                break
            with open(peduncle_mega_val / f"{img_path.stem}_aug_{i}.txt", 'w') as f:
                f.write(annotations)
            cv2.imwrite(str(peduncle_mega_val / f"{img_path.stem}_aug_{i}.png"), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            # Visualize transformed image
            # fig, ax = plt.subplots(figsize=(10,10))
            # ax.imshow(aug_image)

            
            # roi_peduncle_poly = []

            # if annotations is None:
            #     continue
            # for line in annotations.split('\n'):
            #     y = np.array(line.split(" ")[1:]).astype(np.float32)
            #     if line.split(" ")[0] == '0':
            #         roi_peduncle_y = np.array(line.split(" ")[1:]).astype(np.float32)
            #         roi_peduncle_poly.append(roi_peduncle_y)

            # for poly in roi_peduncle_poly:
            #     scaled_poly = poly.reshape((-1,2)) * aug_image.shape[0:2][::-1]
            #     p = patches.Polygon(scaled_poly, facecolor = 'green', edgecolor = 'g',  fill = True, alpha=0.5)

            #     ax.add_patch(p)

            # plt.show()
            # plt.close()

