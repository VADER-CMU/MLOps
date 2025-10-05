# Data collection pipeline

ROS bag -> Images

# How to collect a ROS bag

Install docker

1. Build and run (`build_docker.sh` and `run_docker.sh`) the VADER Docker container: https://github.com/VADER-CMU/vader_docker
2. In your `docker_ws` create a catkin workspace `mkdir -p ~/catkin_ws/src` and cd into it `cd ~/catkin_ws/src/`
3. Download the camera software using `git clone https://github.com/rjwb1/realsense-ros`
4. `cd ../` In the `catkin_ws` folder run `catkin_make`. Further, source your setup file using `source devel/setup.bash`
5. Run the camera node using `roslaunch realsense2_camera rs_camera.launch color_fps:=10 depth_fps:=10`. You may change the FPS
6. Record the bag file using `rosbag record -o <filename prefix> /camera/color/image_raw /camera/depth/image_rect_raw /camera/color/camera_info /camera/depth/camera_info /tf /tf_static`
7. You may visualize the feed using `rosrun rqt_image_view rqt_image_view`
8. Ctrl+C in the rosbag record terminal will save the rosbag

# How to extract images

Run the extraction script `image_extracter.py --bag_file <bag filename> --output_dir <output_dir>`
