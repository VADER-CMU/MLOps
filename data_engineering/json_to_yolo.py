"""
Convert VIA/COCO format JSON segmentation annotations to YOLO segmentation format.
Classes: pepper (0), peduncle (1)
"""

import json
import argparse
from pathlib import Path
from PIL import Image
import sys


def get_class_id(region_attributes):
    """Map region attributes to class IDs."""
    # Check if 'Type' field exists
    if 'Type' in region_attributes:
        type_val = region_attributes['Type'].lower()
        if type_val == 'fruit':
            return 0  # pepper
        elif type_val == 'peduncle':
            return 1  # peduncle
    
    # Fallback: check for direct class names
    if 'class' in region_attributes:
        class_val = region_attributes['class'].lower()
        if 'pepper' in class_val or 'fruit' in class_val:
            return 0
        elif 'peduncle' in class_val:
            return 1
    
    # Default to pepper if unclear
    return 0


def normalize_polygon(all_points_x, all_points_y, img_width, img_height):
    """Normalize polygon coordinates to 0-1 range."""
    normalized_coords = []
    
    for x, y in zip(all_points_x, all_points_y):
        norm_x = x / img_width
        norm_y = y / img_height
        
        # Clamp to [0, 1] to avoid issues
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        normalized_coords.extend([norm_x, norm_y])
    
    return normalized_coords


def convert_via_to_yolo(json_path, output_dir, image_dir=None):
    """
    Convert VIA format JSON to YOLO segmentation format.
    
    Args:
        json_path: Path to VIA format JSON file
        output_dir: Directory to save YOLO format txt files
        image_dir: Optional directory containing images (to get dimensions)
    """
    # Load JSON
    print(f"Loading JSON from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image metadata
    if '_via_img_metadata' in data:
        img_metadata = data['_via_img_metadata']
    else:
        print("Error: '_via_img_metadata' not found in JSON")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(img_metadata)} images...")
    
    converted_count = 0
    skipped_count = 0
    
    for img_key, img_data in img_metadata.items():
        filename = img_data.get('filename', '')
        if not filename:
            print(f"Warning: No filename for key {img_key}, skipping...")
            skipped_count += 1
            continue
        
        # Get image dimensions
        if image_dir:
            img_path = Path(image_dir) / filename
            if img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    print(f"Warning: Could not load image {filename}: {e}")
                    skipped_count += 1
                    continue
            else:
                print(f"Warning: Image {filename} not found in {image_dir}, skipping...")
                skipped_count += 1
                continue
        else:
            # Try to get size from JSON (some formats include it)
            if 'size' in img_data:
                # Size field usually contains file size, not dimensions
                # We need actual image dimensions, so this won't work
                print(f"Warning: No image_dir provided and dimensions not in JSON for {filename}")
                print("Please provide --image_dir argument")
                sys.exit(1)
        
        # Process regions (annotations)
        regions = img_data.get('regions', [])
        if not regions:
            # Create empty label file for images without annotations
            output_filename = Path(filename).stem + '.txt'
            output_path = output_dir / output_filename
            output_path.write_text('')
            converted_count += 1
            continue
        
        annotations = []
        
        for region in regions:
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            
            # Only process polygon annotations
            if shape_attrs.get('name') != 'polygon':
                continue
            
            all_points_x = shape_attrs.get('all_points_x', [])
            all_points_y = shape_attrs.get('all_points_y', [])
            
            if not all_points_x or not all_points_y:
                continue
            
            if len(all_points_x) != len(all_points_y):
                print(f"Warning: Mismatched point counts in {filename}, skipping region")
                continue
            
            # Get class ID
            class_id = get_class_id(region_attrs)
            
            # Normalize coordinates
            normalized_coords = normalize_polygon(
                all_points_x, all_points_y, img_width, img_height
            )
            # Calculate polygon area
            def polygon_area(x_coords, y_coords):
                """Calculate the area of a polygon using the Shoelace formula"""
                n = len(x_coords)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += x_coords[i] * y_coords[j]
                    area -= y_coords[i] * x_coords[j]
                area = abs(area) / 2.0
                return area

            # Calculate relative area (0-1)
            polygon_rel_area = polygon_area(
                [x / img_width for x in all_points_x],
                [y / img_height for y in all_points_y]
            )
            print(f"Polygon relative area for {filename}: {polygon_rel_area:.6f}")

            # Set minimum area threshold (adjust as needed)
            min_area_threshold = 0.007  # 0.01% of image area

            # Skip small polygons
            if class_id == 0 and polygon_rel_area < min_area_threshold:  # Only for fruit/pepper class
                print(f"Skipping small fruit annotation in {filename}, area: {polygon_rel_area:.6f}")
                continue
            
            # Format: class_id x1 y1 x2 y2 ... xn yn
            annotation_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_coords)
            annotations.append(annotation_line)
        
        # Write to YOLO format txt file
        output_filename = Path(filename).stem + '.txt'
        output_path = output_dir / output_filename
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        converted_count += 1
        
        if converted_count % 100 == 0:
            print(f"Converted {converted_count} images...")
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count}")
    # print(f"Skipped: {skipped_count}")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert VIA/COCO JSON segmentation to YOLO format'
    )
    parser.add_argument(
        '--json_file',
        type=str,
        help='Path to input JSON file (VIA/COCO format)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for YOLO format txt files'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory containing the images (required to get image dimensions)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.json_file).exists():
        print(f"Error: JSON file not found: {args.json_file}")
        sys.exit(1)
    
    if args.image_dir and not Path(args.image_dir).exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Perform conversion
    convert_via_to_yolo(args.json_file, args.output_dir, args.image_dir)


if __name__ == '__main__':
    main()