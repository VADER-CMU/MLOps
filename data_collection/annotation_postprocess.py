import shutil
import numpy as np
import cv2
import json
import argparse
import pathlib
import matplotlib.pyplot as plt
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
            if 'pepper' in cls['name'].lower():
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

if __name__ == "__main__":

    # argparse mask folder and output folder
    parser = argparse.ArgumentParser(description="Process segmentation masks")
    parser.add_argument("--masks_folder", type=str, required=True, help="Path to the folder containing mask images")
    args = parser.parse_args()

    mask_folder = pathlib.Path(args.masks_folder)
    output_folder = pathlib.Path(args.masks_folder).parent / "yolo_dataset"
    output_folder.mkdir(parents=True, exist_ok=True)
    mask_files = list(mask_folder.glob("*.png"))
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        yolo_annotations = mask_to_yolo_format(mask)
        output_file = output_folder / (mask_file.stem + ".txt")
        with open(output_file, 'w') as f:
            f.write(yolo_annotations)

    
    # copy images from images_folder to output_folder
    images_folder = mask_folder.parent / "images"
    image_files = list(images_folder.glob("*.png"))
    for image_file in tqdm(image_files, desc="Copying images"):
        shutil.copy(image_file, output_folder / image_file.name)
