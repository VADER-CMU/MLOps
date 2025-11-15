import shutil
import numpy as np
import cv2
import random
from shutil import copy
from pathlib import Path

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import cv2

import albumentations as A
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

PEDUNCLE_IMAGE_SIZE = 640

def extract_peduncle_roi(image, yolo_annotations):
    height = image.shape[0]
    width = image.shape[1]
    fruit_poly = []
    peduncle_poly = []
    cropped_image = None

    cropped_images = []
    annotations = []

    for line in yolo_annotations.split("\n"):
        # y = np.array(line.split(" ")[1:]).astype(np.float32)
        if line.split(" ")[0] == '0':
            fruit_y = np.array(line.split(" ")[1:]).astype(np.float32)
            fruit_poly.append(fruit_y)
        if line.split(" ")[0] == '1':
            peduncle_y = np.array(line.split(" ")[1:]).astype(np.float32)
            peduncle_poly.append(peduncle_y)

    

    for poly in fruit_poly:
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
        
        cropped_image = (cv2.resize(cropped_image, (PEDUNCLE_IMAGE_SIZE, PEDUNCLE_IMAGE_SIZE))*255).astype(np.uint8)
        roi_peduncle_poly = None

        for poly in peduncle_poly:
            scaled_poly = poly.reshape((-1,2)) * image.shape[0:2][::-1]

            if (scaled_poly[:, 0].min() >= roi_x_min and scaled_poly[:, 0].max() <= roi_x_max and
                scaled_poly[:, 1].min() >= roi_y_min and scaled_poly[:, 1].max() <= roi_y_max):

                roi_peduncle_poly = scaled_poly
                break

        new_annotation = []
        if roi_peduncle_poly is not None:
            for point in roi_peduncle_poly.reshape(-1, 2):
                # Normalize coordinates to [0, 1]
                x_norm = (point[0] - roi_x_min) / (roi_x_max - roi_x_min)
                y_norm = (point[1] - roi_y_min) / (roi_y_max - roi_y_min)
                
                # Add to YOLO line
                new_annotation.append(f"{x_norm:.6f} {y_norm:.6f}")

        else:
            continue  # No peduncle in this ROI, skip to next fruit

        new_annotation_line = "0 " + " ".join(new_annotation)
        annotations.append(new_annotation_line)
        cropped_images.append(cropped_image)

    return cropped_images, annotations



# Define the augmentation pipeline
def create_train_augmentation_pipeline():
    transform = A.Compose([
        # Apply random rotation (up to 60 degrees) and translation
        # A.PadIfNeeded(min_height=1000, min_width=1000, border_mode=cv2.BORDER_CONSTANT, value=0),

        # Apply random rotation up to 45 degrees
        A.HorizontalFlip(p=0.5),
        # Always rotate by 90 degrees
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(90, 90), p=0.8),
        

        # A.Rotate(limit=40, p=0.3),  # p=1.0 ensures rotation is always applied[3]
        A.Affine(scale=(1, 1.1), p=0.3 ),
        A.RandomBrightnessContrast(
        brightness_limit=0.3,     # Handle darker/brighter conditions
        contrast_limit=0.2,
        p=0.4
    ),
    A.RandomGamma(
        gamma_limit=(70, 110),    # Gamma correction for different exposures
        p=0.4
    ),

    ])
    return transform

def create_val_augmentation_pipeline():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(90, 90), p=0.8),
    ])
    return transform


def extract_annotations(txt_file):
    annotations = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            # Segmentation coordinates (x1, y1, x2, y2, ...)
            coords = [float(x) for x in parts[1:]]
            annotations.append({'class_id': class_id, 'coords': coords})
    return annotations

def yolo_to_mask(annotations, image_shape):
    height, width = image_shape[0], image_shape[1]
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        poly = np.array(ann['coords']).reshape(-1, 2)
        poly[:, 0] *= width
        poly[:, 1] *= height
        cv2.fillPoly(mask, [poly.astype(np.int32)], color=1)
    return mask

def mask_to_yolo_format(mask):
    """
    Convert an instance segmentation mask to YOLO format
    
    Args:
        mask: numpy array where each unique value > 0 represents an instance
        
    Returns:
        string of YOLO format annotations (class_id, x1, y1, x2, y2, ...)
    """
    height, width = mask.shape
    yolo_annotations = []
    
    # Find unique instance IDs (excluding background 0)
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    for instance_id in instance_ids:
        # Create binary mask for this instance
        instance_mask = (mask == instance_id).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Need at least 3 points to define a polygon
        if len(approx) < 3:
            continue

        
        yolo_line = f"{0}"
                
        for point in approx.reshape(-1, 2):
            # Normalize coordinates to [0, 1]
            x_norm = point[0] / width
            y_norm = point[1] / height
            
            # Add to YOLO line
            yolo_line += f" {x_norm:.6f} {y_norm:.6f}"
        
        yolo_annotations.append(yolo_line)
        
    yolo_string = "\n".join(yolo_annotations)
    

    return yolo_string

def shuffle_split():
    yolo_annotations_raw = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/peduncle_mega/raw")

    # Create a list of all image files
    image_files = list(Path(yolo_annotations_raw).glob("*.png"))

    total_images = len(image_files)

    # print(len(image_files)) #1080
    # Shuffle the image files
    random.shuffle(image_files)

    # Split into train, validation, and test sets
    valid_files = image_files[0:total_images//5]
    train_files = image_files[total_images//5:]

    # Define output directories
    train_dir = "train"
    valid_dir = "valid"

    # Create directories if they don't exist
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(valid_dir).mkdir(parents=True, exist_ok=True)

    # copy the files from raw/images into correspoding train valid and test folders. Move the masks too

    for file_set, output_dir in zip(
        [train_files, valid_files],
        [train_dir, valid_dir]
    ):
        for file_path in file_set:
            # Copy image file
            dest_image_path = Path(output_dir) / file_path.name
            copy(file_path, dest_image_path)

            # Copy corresponding mask file
            mask_file_path = file_path.with_suffix('.txt')
            if mask_file_path.exists():
                copy(mask_file_path, dest_image_path.with_suffix('.txt'))


def run_augmentation(visualize=False):
    peduncle_mega = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/peduncle_mega/train_raw")
    peduncle_mega_train = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/peduncle_mega/train")
    img_list = list(peduncle_mega.glob("*.png"))

    num_augmentations = 5

    for img_path in tqdm(img_list):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read annotations
        annotations = extract_annotations(str(img_path.with_suffix('.txt')))


        for i in range(num_augmentations):
            transform = create_train_augmentation_pipeline()

            # Apply augmentation
            transformed = transform(image=image, masks=[yolo_to_mask(annotations, image.shape)])

            aug_image = transformed['image']
            aug_mask = transformed['masks'][0]

            # Save augmented images and masks
            cv2.imwrite(str(peduncle_mega_train / f"{img_path.stem}_aug_{i}.png"), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            with open(peduncle_mega_train / f"{img_path.stem}_aug_{i}.txt", 'w') as f:
                f.write(mask_to_yolo_format(aug_mask))
            # print(f"Saved: {img_path.stem}")
            # Visualize transformed image
            if visualize:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(aug_image)

                annotation_line = mask_to_yolo_format(aug_mask)
                roi_peduncle_poly = []

                for line in annotation_line.split("\n"):
                # y = np.array(line.split(" ")[1:]).astype(np.float32)
                    if line.split(" ")[0] == '0':
                        roi_peduncle_y = np.array(line.split(" ")[1:]).astype(np.float32)
                        roi_peduncle_poly.append(roi_peduncle_y)

                for poly in roi_peduncle_poly:
                    scaled_poly = poly.reshape((-1,2)) * aug_image.shape[0:2][::-1]
                    p = Polygon(scaled_poly, facecolor = 'green', edgecolor = 'g',  fill = True, alpha=0.5)

                    ax.add_patch(p)

                plt.savefig(f"{peduncle_mega.parent}/augmented_images/{img_path.stem}_aug_{i}.png")
                plt.close()

def run_val_augmentation(visualize=False):
    peduncle_mega = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/peduncle_mega/valid_raw")
    peduncle_mega_val = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/peduncle_mega/valid")
    img_list = list(peduncle_mega.glob("*.png"))

    num_augmentations = 2

    for img_path in tqdm(img_list):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read annotations
        annotations = extract_annotations(str(img_path.with_suffix('.txt')))


        for i in range(num_augmentations):
            transform = create_val_augmentation_pipeline()

            # Apply augmentation
            transformed = transform(image=image, masks=[yolo_to_mask(annotations, image.shape)])

            aug_image = transformed['image']
            aug_mask = transformed['masks'][0]

            # Save augmented images and masks
            cv2.imwrite(str(peduncle_mega_val / f"{img_path.stem}_aug_{i}.png"), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            with open(peduncle_mega_val / f"{img_path.stem}_aug_{i}.txt", 'w') as f:
                f.write(mask_to_yolo_format(aug_mask))
            # print(f"Saved: {img_path.stem}")
            # Visualize transformed image
            if visualize:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(aug_image)

                annotation_line = mask_to_yolo_format(aug_mask)
                roi_peduncle_poly = []

                for line in annotation_line.split("\n"):
                # y = np.array(line.split(" ")[1:]).astype(np.float32)
                    if line.split(" ")[0] == '0':
                        roi_peduncle_y = np.array(line.split(" ")[1:]).astype(np.float32)
                        roi_peduncle_poly.append(roi_peduncle_y)

                for poly in roi_peduncle_poly:
                    scaled_poly = poly.reshape((-1,2)) * aug_image.shape[0:2][::-1]
                    p = Polygon(scaled_poly, facecolor = 'green', edgecolor = 'g',  fill = True, alpha=0.5)

                    ax.add_patch(p)

                plt.savefig(f"{peduncle_mega.parent}/augmented_images/{img_path.stem}_aug_{i}.png")
                plt.close()


if __name__ == "__main__":
    # img_path = "/home/kshitij/Documents/Bell Pepper/dataset-collection/dataset_620_red_yellow_cart_only/_Color_1607625615060.01098632812500.png"
    # image = plt.imread(img_path)
    # annotation_path = "/home/kshitij/Documents/Bell Pepper/dataset-collection/labels/_Color_1607625615060.01098632812500.txt"

    # yolo_annotations = open(annotation_path).read()

    # cropped_image, new_annotation_line = extract_peduncle_roi(image, yolo_annotations)
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.imshow(cropped_image)
    # roi_peduncle_poly = []

    # for line in new_annotation_line.split("\n"):
    # # y = np.array(line.split(" ")[1:]).astype(np.float32)
    #     if line.split(" ")[0] == '0':
    #         roi_peduncle_y = np.array(line.split(" ")[1:]).astype(np.float32)
    #         roi_peduncle_poly.append(roi_peduncle_y)

    # for poly in roi_peduncle_poly:
    #     scaled_poly = poly.reshape((-1,2)) * cropped_image.shape[0:2][::-1]
    #     p = Polygon(scaled_poly, facecolor = 'green', edgecolor = 'g',  fill = True, alpha=0.5)

    #     ax.add_patch(p)

    # plt.show()

    # img_dir = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/dataset_620_red_yellow_cart_only")
    # label_dir = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/labels_p_620")
    # output_dir = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/peduncle_620")

    # # run the extraction on all images
    # img_list = list(img_dir.glob("*.png"))

    # counter = 0

    # for img_path in img_list:
    #     image = plt.imread(img_path)
    #     annotation_path = label_dir / (img_path.stem + ".txt")

    #     yolo_annotations = open(annotation_path).read()

    #     cropped_images, annotation_lines = extract_peduncle_roi(image, yolo_annotations)


    #     for cropped_image, annotation_line in zip(cropped_images, annotation_lines):

    #         # save cropped image and new annotation with randomized names
    #         rand_name = f"peduncle_{counter}"
    #         counter += 1
    #         cv2.imwrite(str(output_dir / f'{rand_name}.png'), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))  # convert RGB to BGR for cv2
    #         with open(output_dir / f'{rand_name}.txt', 'w') as f:
    #             f.write(annotation_line)
    #         fig, ax = plt.subplots(figsize=(10,10))
    #         ax.imshow(cropped_image)
    #         roi_peduncle_poly = []

    #         for line in annotation_line.split("\n"):
    #         # y = np.array(line.split(" ")[1:]).astype(np.float32)
    #             if line.split(" ")[0] == '0':
    #                 roi_peduncle_y = np.array(line.split(" ")[1:]).astype(np.float32)
    #                 roi_peduncle_poly.append(roi_peduncle_y)

    #         for poly in roi_peduncle_poly:
    #             scaled_poly = poly.reshape((-1,2)) * cropped_image.shape[0:2][::-1]
    #             p = Polygon(scaled_poly, facecolor = 'green', edgecolor = 'g',  fill = True, alpha=0.5)

    #             ax.add_patch(p)

    #         plt.savefig(output_dir / f'{rand_name}_plot.png')
    #         plt.close()

    # peduncle_mega = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/peduncle_mega/raw")
    # peduncle_620 = Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/peduncle_620")

    # # copy all images and labels from peduncle_620 to peduncle_mega
    # img_list = list(peduncle_620.glob("*.png"))
    # label_list = list(peduncle_620.glob("*.txt"))

    # for img_path in img_list:
    #     if "plot" not in img_path.name:
    #         shutil.copy(img_path, peduncle_mega / img_path.name)
    #         label_path = img_path.with_suffix('.txt')
    #         shutil.copy(label_path, peduncle_mega / label_path.name)

    # shuffle_split()
    # run_augmentation()
    run_val_augmentation()
   
