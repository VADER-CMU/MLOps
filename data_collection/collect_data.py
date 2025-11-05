import shutil
import pyrealsense2 as rs
import numpy as np
import cv2
import json
from datetime import datetime
from pathlib import Path
import time

class RealSenseCollector:
    def __init__(self, width=640, height=480, fps=10, output_dir="realsense_data"):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir
        
        # Create output directories
        self.rgb_dir = Path(output_dir) / "images"
        self.depth_dir = Path(output_dir) / "depth"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Exposure and gain settings to cycle through
        self.exposure_values = [5000, 10000, 20000, 33000]  # in microseconds
        self.gain_values = [16, 32, 64, 128]  # gain values
        self.current_setting_idx = 0
        self.frames_per_setting = 17
        
    def save_intrinsics(self):
        """Save camera intrinsics to JSON file before starting collection"""
        profile = self.pipeline.get_active_profile()
        
        # Get color stream intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Get depth stream intrinsics
        depth_stream = profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        intrinsics_data = {
            "color": {
                "width": color_intrinsics.width,
                "height": color_intrinsics.height,
                "ppx": color_intrinsics.ppx,
                "ppy": color_intrinsics.ppy,
                "fx": color_intrinsics.fx,
                "fy": color_intrinsics.fy,
                "model": str(color_intrinsics.model),
                "coeffs": color_intrinsics.coeffs
            },
            "depth": {
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "ppx": depth_intrinsics.ppx,
                "ppy": depth_intrinsics.ppy,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "model": str(depth_intrinsics.model),
                "coeffs": depth_intrinsics.coeffs
            }
        }
        
        intrinsics_path = Path(self.output_dir) / "intrinsics.json"
        with open(intrinsics_path, 'w') as f:
            json.dump(intrinsics_data, f, indent=4)
        
        print(f"✓ Intrinsics saved to {intrinsics_path}")
        return intrinsics_data
    
    @staticmethod
    def set_exposure_gain(profile, exposure, gain):
        """Set exposure and gain for the color sensor"""
        # color_sensor = profile.get_device().query_sensors()[1]
        profile.get_device().sensors[0].set_option(rs.option.exposure, exposure)
        profile.get_device().sensors[0].set_option(rs.option.gain, gain)
        # color_sensor.set_option(rs.option.gain, gain)
        print(f"✓ Set exposure={exposure}μs, gain={gain}")
    
    def collect_data(self, duration_seconds=60):
        """Collect RGB and depth data for specified duration"""
        try:
            # Start pipeline
            profile = self.pipeline.start(self.config)
            profile.get_device().sensors[0].set_option(rs.option.enable_auto_exposure, 0)
            print(f"✓ Pipeline started at {self.width}x{self.height} @ {self.fps} FPS")
            
            # Wait for camera to stabilize
            for _ in range(30):
                self.pipeline.wait_for_frames()
            
            # Save intrinsics
            self.save_intrinsics()
            
            # Initialize frame counter and metadata
            frame_count = 0
            metadata = []
            
            # Calculate target frame time
            target_frame_time = 1.0 / self.fps
            start_time = time.time()
            
            print(f"\n🎥 Starting data collection for {duration_seconds} seconds...")
            print(f"📁 Saving to: {self.output_dir}")
            print(f"Press Ctrl+C to stop early\n")
            
            while time.time() - start_time < duration_seconds:
                loop_start = time.time()
                
                # Check if we need to change exposure/gain
                if frame_count % self.frames_per_setting == 0:
                    exposure = self.exposure_values[self.current_setting_idx]
                    gain = self.gain_values[self.current_setting_idx]
                    self.set_exposure_gain(profile, exposure, gain)
                    self.current_setting_idx = (self.current_setting_idx + 1) % len(self.exposure_values)
                
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Save frames
                timestamp = time.time()
                rgb_filename = f"rgb_{frame_count:06d}.png"
                depth_filename = f"depth_{frame_count:06d}.png"
                
                if frame_count % 5 == 0:
                    cv2.imwrite(str(self.rgb_dir / rgb_filename), color_image)
                    cv2.imwrite(str(self.depth_dir / depth_filename), depth_image)
                
                # Get current sensor settings
                try:
                    color_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
                    current_exposure = color_sensor.get_option(rs.option.exposure)
                    current_gain = color_sensor.get_option(rs.option.gain)
                except:
                    current_exposure = None
                    current_gain = None
                
                # Store metadata
                metadata.append({
                    "frame_id": frame_count,
                    "timestamp": timestamp,
                    "rgb_file": rgb_filename,
                    "depth_file": depth_filename,
                    "exposure": current_exposure,
                    "gain": current_gain
                })
                
                frame_count += 1
                
                # Progress update
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"Collected {frame_count//5} frames in {elapsed:.1f}s")
                
                # Maintain target FPS
                elapsed = time.time() - loop_start
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
            
            # Save metadata
            metadata_path = Path(self.output_dir) / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n✅ Collection complete!")
            print(f"📊 Total frames collected: {frame_count}")
            print(f"📁 Data saved to: {self.output_dir}")
            print(f"📄 Metadata saved to: {metadata_path}")
            
        except KeyboardInterrupt:
            print("\n⚠ Collection interrupted by user")
        
        finally:
            self.pipeline.stop()
            print("✓ Pipeline stopped")
        classes_src = 'classes.json'
        classes_dst = Path(self.output_dir) / 'classes.json'
        if Path(classes_src).exists():
            shutil.copy(classes_src, classes_dst)
            print(f"Copied {classes_src} to {classes_dst}")
        else:
            print(f"File {classes_src} not found, skipping copy.")

if __name__ == "__main__":
    # Configuration
    WIDTH = 640
    HEIGHT = 480
    FPS = 5
    DURATION = 3600  # seconds
    OUTPUT_DIR = f"realsense_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create collector and start
    collector = RealSenseCollector(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        output_dir=OUTPUT_DIR
    )
    
    collector.collect_data(duration_seconds=DURATION)