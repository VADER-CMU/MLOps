"""
Extract RGB and Depth images from ROS 1 bag file without ROS installation.
Uses rosbags library for pure Python deserialization.
"""

from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.image import message_to_cvimage
import cv2
import os
import argparse
import shutil

def extract_images_from_rosbag(bag_path, rgb_topic, depth_topic, output_dir):
    """
    Extract RGB and depth images from a ROS bag file without ROS installation.
    
    Args:
        bag_path (str): Path to the rosbag file
        rgb_topic (str): Topic name for RGB images
        depth_topic (str): Topic name for depth images
        output_dir (str): Directory to save extracted images
    """
    # Create output directories
    rgb_dir = os.path.join(output_dir, 'images')
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # Convert bag_path to Path object
    bagpath = Path(bag_path)
    
    # Create type store for ROS1
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    rgb_count = 0
    depth_count = 0
    
    print(f"Opening bag file: {bag_path}")
    print(f"RGB topic: {rgb_topic}")
    print(f"Depth topic: {depth_topic}")
    print(f"Output directory: {output_dir}\n")
    
    # Open bag file with AnyReader (supports both ROS1 and ROS2)
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        # Print available topics
        print("Available topics in bag:")
        for connection in reader.connections:
            print(f"  {connection.topic} ({connection.msgtype})")
        print()
        
        # Get connections for our topics
        rgb_connections = [x for x in reader.connections if x.topic == rgb_topic]
        depth_connections = [x for x in reader.connections if x.topic == depth_topic]
        
        if not rgb_connections:
            print(f"Warning: RGB topic '{rgb_topic}' not found in bag file")
        if not depth_connections:
            print(f"Warning: Depth topic '{depth_topic}' not found in bag file")
        
        target_connections = rgb_connections + depth_connections
        
        # Read and process messages
        for connection, timestamp, rawdata in reader.messages(connections=target_connections):
            # Deserialize the message
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            if connection.topic == rgb_topic:
                try:
                    # Convert ROS Image message to OpenCV image (BGR format)
                    cv_image = message_to_cvimage(msg, 'bgr8')
                    
                    # Create filename with timestamp
                    timestamp_str = f"{timestamp // 1000000000:010d}_{timestamp % 1000000000:09d}"
                    rgb_filename = os.path.join(rgb_dir, f'rgb_{timestamp_str}_{rgb_count:06d}.png')
                    
                    # Save image
                    # cv2.imwrite(rgb_filename, cv_image)
                    rgb_count += 1
                    
                    if rgb_count % 20 == 0:
                        cv2.imwrite(rgb_filename, cv_image)
                        print(f"Saved {rgb_count} RGB images...")
                        
                except Exception as e:
                    print(f"Error processing RGB image {rgb_count}: {e}")
            
            elif connection.topic == depth_topic:
                try:
                    # Convert ROS Depth Image message to OpenCV image
                    # Keep original encoding (usually 16UC1 or 32FC1)
                    cv_depth_image = message_to_cvimage(msg)
                    
                    # Create filename with timestamp
                    timestamp_str = f"{timestamp // 1000000000:010d}_{timestamp % 1000000000:09d}"
                    depth_filename = os.path.join(depth_dir, f'depth_{timestamp_str}_{depth_count:06d}.png')
                    
                    # Save depth image
                    # If depth is float32, convert to uint16 (millimeters)
                    if cv_depth_image.dtype == 'float32':
                        cv_depth_image = (cv_depth_image * 1000).astype('uint16')
                    
                    # cv2.imwrite(depth_filename, cv_depth_image)
                    depth_count += 1
                    
                    if depth_count % 20 == 0:
                        cv2.imwrite(depth_filename, cv_depth_image)
                        print(f"Saved {depth_count} depth images...")
                        
                except Exception as e:
                    print(f"Error processing depth image {depth_count}: {e}")
    
    print(f"\nExtraction completed!")
    print(f"Total RGB images saved: {rgb_count}")
    print(f"Total depth images saved: {depth_count}")
    print(f"RGB images location: {rgb_dir}")
    print(f"Depth images location: {depth_dir}")

    classes_src = 'classes.json'
    classes_dst = os.path.join(output_dir, 'classes.json')
    if os.path.exists(classes_src):
        shutil.copy(classes_src, classes_dst)
        print(f"Copied {classes_src} to {classes_dst}")
    else:
        print(f"File {classes_src} not found, skipping copy.")

def main():
    parser = argparse.ArgumentParser(
        description='Extract RGB and depth images from ROS 1 bag file (no ROS installation required)'
    )
    parser.add_argument('--bag_file', help='Path to the ROS bag file')
    parser.add_argument('--rgb_topic', default='/camera/color/image_raw',
                       help='RGB image topic (default: /camera/color/image_raw)')
    parser.add_argument('--depth_topic', default='/camera/depth/image_rect_raw',
                       help='Depth image topic (default: /camera/depth/image_rect_raw)')
    parser.add_argument('--output_dir', default='./extracted_images',
                       help='Output directory for images (default: ./extracted_images)')
    
    args = parser.parse_args()
    
    # Check if bag file exists
    if not os.path.exists(args.bag_file):
        print(f"Error: Bag file '{args.bag_file}' does not exist")
        return
    
    extract_images_from_rosbag(args.bag_file, args.rgb_topic, args.depth_topic, args.output_dir)

if __name__ == '__main__':
    main()
