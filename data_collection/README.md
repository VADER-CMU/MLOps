# Data collection & Annotation pipeline

# Data collection

ROS bag -> Images

## How to collect a ROS bag

Install docker

1. Build and run (`build_docker.sh` and `run_docker.sh`) the VADER Docker container: https://github.com/VADER-CMU/vader_docker
2. In your `docker_ws` create a catkin workspace `mkdir -p catkin_ws/src` and cd into it `cd catkin_ws/src/`
3. Download the camera software using `git clone https://github.com/VADER-CMU/realsense-ros.git`
4. In the docker container run `catkin_make` within the `catkin_ws`. Further, source your setup file using `source devel/setup.bash`
5. Run the camera node using `roslaunch realsense2_camera rs_camera.launch color_fps:=10 depth_fps:=10`. You may change the FPS
6. Record the bag file using `rosbag record -o <filename prefix> /camera/color/image_raw /camera/depth/image_rect_raw /camera/color/camera_info /camera/depth/camera_info /tf /tf_static`
7. You may visualize the feed using `rosrun rqt_image_view rqt_image_view`
8. Ctrl+C in the rosbag record terminal will save the rosbag

## How to extract images

Create a **Python 3.11** `.venv` or a conda env and install `pip install opencv-python rosbags rosbags-image` 

Run the extraction script `python image_extracter.py --bag_file <bag filename> --output_dir <output_dir>`

# Data collection (Option 2)

Use `collect_data.py` to collect data. Requirements: `numpy opencv-python pyrealsense2 json pathlib`

```bash
python collect_data.py
```

# Annotation

Use a **Python 3.11** `.venv` or a conda env and install SAM in the env using `pip install git+https://github.com/facebookresearch/segment-anything.git`

## Run SAMAT

1. Do `git submodule update --init --recursive` to init the samat repo
2. Install SAMAT in the env: run `python -m pip install -e .` in the `samat` folder.
3. Download SAM VIT_H weights `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`
4. Edit `samat/config.toml` with the dataset (`output_dir`) and path to the weights
5. In `samat` folder, run `python scripts/preprocess_dataset.py`. This creates the `sam` folder with SAM masks
6. Run the annotation tool `python .` 
7. Follow the shortcuts given here: https://github.com/Divelix/samat/tree/9bd13b742accc4d804d962bb07e7ad31cbbd9a8c?tab=readme-ov-file#shortcuts
