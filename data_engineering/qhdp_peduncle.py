import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pathlib

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

    peduncle_folder = pathlib.Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/qhdp2020/Peduncle_QHDP/train/peduncle_imgs")
    cropped_folder = pathlib.Path("/home/kshitij/Documents/Bell Pepper/dataset-collection/qhdp2020/Peduncle_QHDP/train/cropped_imgs")

    xml_list = list(peduncle_folder.glob("*.xml"))
    for xml_file in xml_list:
        tree = ET.parse(str(xml_file))
        root = tree.getroot()
        image_name_from_xml = xml_file.parent / root.find('filename').text

        image_file = image_name_from_xml

        cropped_regions = extract_cropped_regions(xml_file, image_file)
        counter = 0

        for img in cropped_regions:
            cv2.imwrite(os.path.join(cropped_folder, os.path.basename(xml_file).replace('.xml', f'_peduncle{counter}.png')), img)
            counter += 1
